# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Observation-only workload for the HW2 communication report; run under msprof."""

import argparse
import asyncio
import functools
import json
import os
import sys
import time
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TEST_ROOT))
import torch
import torch_npu
from aics.hw2.tp_validation import engine_options, generate_queued

from tokenspeed.runtime.distributed.comm_backend.hccl import HcclBackend
from tokenspeed.runtime.execution.model_executor import ModelExecutor

OUT = Path(os.environ.get("HW2_ANALYSIS_OUT", Path.cwd()))
original_forward = ModelExecutor._forward_step
step = 0


def forward(self, bs, ctx, sampling_info):
    global step
    step += 1
    rank = torch.distributed.get_rank()
    label = f"HW2_forward_rank{rank}_step{step}_bs{ctx.bs}_extend{ctx.num_extends}_tokens{ctx.input_num_tokens}"
    span = torch_npu.npu.mstx.range_start(label)
    started = time.perf_counter_ns()
    result = original_forward(self, bs, ctx, sampling_info)
    torch_npu.npu.mstx.range_end(span)
    row = dict(
        rank=rank,
        step=step,
        bs=ctx.bs,
        extends=ctx.num_extends,
        tokens=ctx.input_num_tokens,
        start_ns=started,
        end_ns=time.perf_counter_ns(),
    )
    with (OUT / f"rank{rank}-forwards.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    return result


ModelExecutor._forward_step = forward

for method_name in ("all_gather_single", "all_reduce"):
    method = getattr(HcclBackend, method_name)

    def annotate(method, name):
        @functools.wraps(method)
        def call(self, *args, **kwargs):
            tensor = args[1] if name == "all_gather_single" else args[0]
            if not isinstance(tensor, torch.Tensor):
                return method(self, *args, **kwargs)
            label = f"HW2_{name}_{str(tensor.dtype)}_{list(tensor.shape)}"
            span = torch_npu.npu.mstx.range_start(label)
            try:
                return method(self, *args, **kwargs)
            finally:
                torch_npu.npu.mstx.range_end(span)

        return call

    setattr(HcclBackend, method_name, annotate(method, method_name))


async def run(engine, ids, batch, sampling):
    if batch == 1:
        return [await engine.async_generate(input_ids=ids, sampling_params=sampling)]
    return await generate_queued(engine, [ids, ids], sampling)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(Path.home() / "models/Qwen3.5-9B"))
    parser.add_argument("--tp", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    global OUT
    OUT = Path(args.output)
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ["HW2_ANALYSIS_OUT"] = str(OUT)
    from transformers import AutoTokenizer

    from tokenspeed.runtime.entrypoints.engine import Engine

    model = args.model
    tp = args.tp
    opts = engine_options(model, tp=tp, enable_prefix_caching=False)
    opts.update(log_level="warning")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    source = tokenizer.encode(
        "春天来了，校园里开满了鲜花。" * 600, add_special_tokens=False
    )
    sampling = dict(temperature=0, max_new_tokens=8, ignore_eos=True)
    engine = Engine(**opts)
    rows = []

    async def workload():
        for length in (128, 2048):
            for batch in (1, 2):
                for phase in ("warmup", "measured"):
                    started = time.perf_counter_ns()
                    results = await run(engine, source[:length], batch, sampling)
                    assert len(results) == batch
                    assert all(len(r["output_ids"]) == 8 for r in results)
                    row = dict(
                        input_length=length,
                        batch=batch,
                        phase=phase,
                        start_ns=started,
                        end_ns=time.perf_counter_ns(),
                        output_ids=[r["output_ids"] for r in results],
                    )
                    rows.append(row)
                    print("WORKLOAD", json.dumps(row), flush=True)

    try:
        asyncio.run(workload())
        (OUT / "workload.json").write_text(
            json.dumps(dict(tp=tp, options=opts, rows=rows), indent=2)
        )
    finally:
        engine.shutdown()
    print("WORKLOAD_PASS", flush=True)


if __name__ == "__main__":
    main()

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

"""Small in-process benchmark for two requests forced into one decode batch."""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if os.environ.get("BATCH_DIAGNOSTICS") == "1":
    from aics.common.measure_hooks import install

    install(enabled=True)


async def queued_batch(engine, input_ids, sampling):
    manager = engine.tokenizer_manager
    if not await manager.pause_scheduler(mode="keep"):
        raise RuntimeError("could not pause scheduler")
    previous = set(manager.rid_to_state)
    task = asyncio.create_task(
        engine.async_generate(input_ids=input_ids, sampling_params=sampling)
    )
    try:
        async with asyncio.timeout(30):
            while len(set(manager.rid_to_state) - previous) < len(input_ids):
                await asyncio.sleep(0.001)
        if not await manager.is_scheduler_paused():
            raise RuntimeError("scheduler resumed before both requests were queued")
    except BaseException:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise
    finally:
        await manager.resume_scheduler()
    return await task


def options(model, tp, max_total_tokens):
    return dict(
        model=model,
        device="npu",
        dtype="bfloat16",
        world_size=tp,
        attn_tp_size=tp,
        dense_tp_size=tp,
        lm_head_tp_size=tp,
        data_parallel_size=1,
        language_model_only=True,
        skip_tokenizer_init=True,
        kv_cache_dtype="auto",
        attention_backend="mha",
        sampling_backend="greedy",
        mamba_ssm_dtype="float32",
        enforce_eager=True,
        disable_prefill_graph=True,
        disable_pdl=True,
        disable_autotune=True,
        enable_prefix_caching=False,
        disable_kvstore=True,
        retraction_snapshot_ratio=0,
        seed=0,
        max_model_len=4096,
        max_num_seqs=2,
        max_total_tokens=max_total_tokens,
        chunked_prefill_size=4096,
        prefix_granularity=128,
        log_level="warning",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, choices=(1, 2), required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--input-lengths", type=int, nargs="+", default=[128, 2048])
    parser.add_argument("--max-total-tokens", type=int, default=16384)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from tokenspeed.runtime.entrypoints.engine import Engine

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    source = tokenizer.encode(
        "春天来了，校园里开满了鲜花。" * 700, add_special_tokens=False
    )
    if args.max_total_tokens < 2 * max(args.input_lengths) + 32:
        parser.error("max-total-tokens must fit two requests and 32 generated tokens")
    engine = Engine(**options(args.model, args.tp, args.max_total_tokens))
    sampling = dict(
        temperature=0,
        max_new_tokens=32,
        ignore_eos=True,
        skip_special_tokens=False,
    )
    rows = []
    try:
        for length in args.input_lengths:
            batch = [source[:length], source[:length]]
            started = time.perf_counter()
            engine.llm.run(queued_batch(engine, batch, sampling))
            warmup_s = time.perf_counter() - started
            print(f"WARMUP input={length} e2e_ms={warmup_s * 1000:.3f}", flush=True)
            values = []
            for repeat in range(args.repeats):
                started = time.perf_counter()
                result = engine.llm.run(queued_batch(engine, batch, sampling))
                elapsed_s = time.perf_counter() - started
                if len(result) != 2 or any(
                    row["meta_info"]["completion_tokens"] != 32 for row in result
                ):
                    raise RuntimeError("incomplete batch result")
                values.append(elapsed_s)
                print(
                    f"RUN input={length} repeat={repeat + 1} "
                    f"E2E_ms={elapsed_s * 1000:.3f} "
                    f"output_tok_s={64 / elapsed_s:.3f}",
                    flush=True,
                )
            row = dict(
                input_length=length,
                batch=2,
                repeats=len(values),
                e2e_ms=statistics.mean(values) * 1000,
                e2e_stdev_ms=(
                    statistics.stdev(values) * 1000 if len(values) > 1 else 0
                ),
                output_tok_s=64 / statistics.mean(values),
            )
            rows.append(row)
            print(f"SUMMARY {json.dumps(row, ensure_ascii=False)}", flush=True)
    finally:
        engine.shutdown()
    Path(args.output).write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

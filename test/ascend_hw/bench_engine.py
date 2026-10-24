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

"""Print warmed-up latency statistics, or separate Event/HBM diagnostics."""

import argparse
import os
import statistics
import sys
import time

sys.dont_write_bytecode = True
from measure_hooks import install
from model_checks import default_model

install(
    enabled=os.environ.get("FORWARD_DIAGNOSTICS") == "1" and sys.argv[1:2] == ["memory"]
)


def measure_stream(stream, *, expected_output, started, clock):
    arrivals = []
    previous = 0
    final = None
    for item in stream:
        now = clock()
        count = item["meta_info"]["completion_tokens"]
        if count > previous:
            assert count == previous + 1, "Expected one streaming arrival per token"
            arrivals.append(now - started)
            previous = count
        final = item
    assert previous == expected_output, (previous, expected_output)
    assert len(arrivals) == expected_output and expected_output > 1
    return dict(
        input_tokens=final["meta_info"]["prompt_tokens"],
        output_tokens=previous,
        ttft_ms=arrivals[0] * 1000,
        decode_ms_per_token=(arrivals[-1] - arrivals[0]) * 1000 / (expected_output - 1),
        e2e_ms=arrivals[-1] * 1000,
    )


def summarize_runs(rows):
    if not rows:
        raise ValueError("At least one measured request is required")
    summary = {}
    for field in ("ttft_ms", "decode_ms_per_token", "e2e_ms"):
        values = [r[field] for r in rows]
        summary[field] = dict(
            mean=statistics.mean(values),
            stdev=statistics.stdev(values) if len(values) > 1 else 0,
            median=statistics.median(values),
            min=min(values),
            max=max(values),
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["benchmark", "memory"])
    parser.add_argument("--model", default=default_model())
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--input-lengths", type=int, nargs="+", default=[128, 2048])
    parser.add_argument("--output-lengths", type=int, nargs="+", default=[32, 128])
    args = parser.parse_args()
    if args.repeats < 1 or min(args.input_lengths) < 1 or min(args.output_lengths) < 2:
        parser.error(
            "Positive input lengths/repeats and at least two output tokens are required"
        )
    if max(args.input_lengths) + max(args.output_lengths) > 4096:
        parser.error("Input plus output length must fit max_model_len=4096")
    os.environ["FORWARD_DIAGNOSTICS"] = "1" if args.mode == "memory" else "0"
    install(enabled=args.mode == "memory")

    from transformers import AutoTokenizer

    from tokenspeed.runtime.entrypoints.engine import Engine

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    source = tokenizer.encode(
        "春天来了，校园里开满了鲜花。" * 600, add_special_tokens=False
    )
    assert len(source) >= max(args.input_lengths)
    engine = Engine(
        model=args.model,
        device="npu",
        dtype="bfloat16",
        world_size=1,
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
        max_total_tokens=8192,
        chunked_prefill_size=4096,
        prefix_granularity=128,
        log_level="warning",
    )

    def request(length, generated):
        started = time.perf_counter()
        stream = engine.generate(
            input_ids=source[:length],
            sampling_params=dict(
                temperature=0,
                max_new_tokens=generated,
                ignore_eos=True,
                skip_special_tokens=False,
            ),
            stream=True,
        )
        row = measure_stream(
            stream, expected_output=generated, started=started, clock=time.perf_counter
        )
        assert row["input_tokens"] == length
        return row

    try:
        print(f'MODE={args.mode} diagnostics={args.mode == "memory"}', flush=True)
        for length in args.input_lengths:
            print(
                f"WARMUP input={length} output={max(args.output_lengths)}", flush=True
            )
            request(length, max(args.output_lengths))
            for generated in args.output_lengths:
                if args.mode == "memory":
                    print(
                        f"DIAGNOSTIC_REQUEST input={length} output={generated}",
                        flush=True,
                    )
                    request(length, generated)
                    continue
                rows = []
                for repeat in range(args.repeats):
                    row = request(length, generated)
                    rows.append(row)
                    print(
                        f"RUN input={length} output={generated} repeat={repeat+1} "
                        f"TTFT_ms={row['ttft_ms']:.3f} TPOT_ms={row['decode_ms_per_token']:.3f}",
                        flush=True,
                    )
                summary = summarize_runs(rows)
                print(
                    f"SUMMARY input={length} output={generated} repeats={len(rows)} "
                    f"TTFT_ms={summary['ttft_ms']['mean']:.3f}+/-{summary['ttft_ms']['stdev']:.3f} "
                    f"TPOT_ms={summary['decode_ms_per_token']['mean']:.3f}+/-{summary['decode_ms_per_token']['stdev']:.3f} "
                    f"E2E_ms={summary['e2e_ms']['mean']:.3f}",
                    flush=True,
                )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()

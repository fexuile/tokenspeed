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

"""Sequential TP acceptance runs; comparison does not import the NPU runtime."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path

# Spawned workers import this module again; forked workers inherit the hook.
if os.environ.get("TOKENSPEED_TP_BATCH_CAPTURE"):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from aics.hw2.batch_capture import install

    install(os.environ["TOKENSPEED_TP_BATCH_CAPTURE"])


def read_batch_evidence(directory, *, tp):
    evidence = {}
    for rank in range(tp):
        path = Path(directory) / f"rank{rank}.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        evidence[str(rank)] = dict(
            paired_decode_steps=sum(
                e["bs"] == 2 and e["num_extends"] == 0 for e in events
            ),
            max_batch_size=max(e["bs"] for e in events),
        )
    return evidence


def require_paired_decode(record):
    evidence = record.get("batch_evidence", {})
    for rank in range(record["tp"]):
        # Two 32-output concurrent batches must share nearly all decode steps.
        if evidence.get(str(rank), {}).get("paired_decode_steps", 0) < 60:
            raise ValueError(f"Rank {rank} lacks two-request decode evidence")


async def generate_queued(engine, input_ids, sampling):
    manager = engine.tokenizer_manager
    if not await manager.pause_scheduler(mode="keep"):
        raise RuntimeError("Could not pause scheduler for concurrent submission")
    previous_rids = set(manager.rid_to_state)
    task = asyncio.create_task(
        engine.async_generate(input_ids=input_ids, sampling_params=sampling)
    )
    try:
        async with asyncio.timeout(30):
            while len(set(manager.rid_to_state) - previous_rids) < len(input_ids):
                await asyncio.sleep(0.001)
        # This control reply follows both submissions on the scheduler channel.
        if not await manager.is_scheduler_paused():
            raise RuntimeError("Scheduler resumed before both requests were queued")
    except BaseException:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise
    finally:
        await manager.resume_scheduler()
    return await task


def model_geometry(config, *, tp):
    text = config.get("text_config", config)
    h = text["hidden_size"]
    q = text["num_attention_heads"]
    kv = text["num_key_value_heads"]
    d = text.get("head_dim", h // q)
    nk, nv = text["linear_num_key_heads"], text["linear_num_value_heads"]
    dk, dv = text["linear_key_head_dim"], text["linear_value_head_dim"]
    intermediate = text["intermediate_size"]
    if tp not in (1, 2) or any(n % tp for n in (q, nk, nv, intermediate)):
        raise ValueError("Model dimensions must support the requested TP width")
    if kv >= tp and kv % tp:
        raise ValueError("KV heads must divide TP, or be replicated")
    k, v = nk * dk, nv * dv
    return dict(
        hidden=h,
        q_heads=q // tp,
        kv_heads=max(1, kv // tp),
        head_dim=d,
        qkv_weight=[(2 * q // tp + 2 * max(1, kv // tp)) * d, h],
        attention_output_weight=[h // tp, q * d],
        gdn_output_weight=[h // tp, v],
        gdn_projection_weight=[(2 * k + 2 * v + 2 * nv) // tp, h],
        convolution_weight=[
            (2 * k + v) // tp,
            1,
            text.get("linear_conv_kernel_dim", 4),
        ],
        convolution_state=[
            (2 * k + v) // tp,
            text.get("linear_conv_kernel_dim", 4) - 1,
        ],
        recurrent_state=[nv // tp, dv, dk],
        gate_up_weight=[2 * intermediate // tp, h],
        down_weight=[h // tp, intermediate],
        output_projection_layout="full_k_output_sharded",
        vocab_size=text["vocab_size"],
        tied_embeddings=text.get(
            "tie_word_embeddings", config.get("tie_word_embeddings", False)
        ),
    )


def checkpoint_identity(model):
    root = Path(model).resolve()
    # Hash tensor files as well as tokenizer/config files: equal paths or config
    # hashes alone cannot establish that two runs used the same weights.
    files = sorted(
        p
        for p in root.iterdir()
        if p.is_file()
        and p.suffix in (".json", ".safetensors", ".bin", ".model", ".jinja")
    )
    if not any(p.suffix in (".safetensors", ".bin") for p in files):
        raise ValueError("No local checkpoint weights found")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        with path.open("rb") as stream:
            digest.update(hashlib.file_digest(stream, "sha256").digest())
    return digest.hexdigest()


def engine_options(model, *, tp, enable_prefix_caching):
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
        enable_prefix_caching=enable_prefix_caching,
        disable_kvstore=True,
        retraction_snapshot_ratio=0,
        seed=0,
        max_model_len=4096,
        max_num_seqs=2,
        max_total_tokens=16384,
        chunked_prefill_size=4096,
        prefix_granularity=128,
    )


def compare_records(actual, reference):
    if not actual["completed"] or not reference["completed"]:
        raise ValueError("Run did not complete its acceptance schedule")
    require_paired_decode(actual)
    require_paired_decode(reference)
    for key in ("checkpoint_sha256", "settings", "sampling"):
        if actual[key] != reference[key]:
            raise ValueError(f"Incompatible run metadata: {key}")
    if not actual["rows"] or len(actual["rows"]) != len(reference["rows"]):
        raise ValueError("Missing comparison cases")
    comparisons = []
    isolated = {}
    for row, expected in zip(actual["rows"], reference["rows"], strict=True):
        if (row["case"], row["phase"], row["input_ids"]) != (
            expected["case"],
            expected["phase"],
            expected["input_ids"],
        ):
            raise ValueError("Input or request schedule differs")
        if len(row["output_ids"]) != actual["sampling"]["max_new_tokens"]:
            raise ValueError("Incomplete decode run")
        key = tuple(row["input_ids"])
        isolated.setdefault(key, row["output_ids"])
        comparisons.append(
            dict(
                case=row["case"],
                phase=row["phase"],
                exact_tokens=row["output_ids"] == expected["output_ids"],
                isolated_tokens=row["output_ids"] == isolated[key],
            )
        )
    return dict(
        passed=all(r["exact_tokens"] and r["isolated_tokens"] for r in comparisons),
        comparisons=comparisons,
    )


def inspect(model, *, tp):
    import tokenspeed_kernel
    import tokenspeed_kernel_npu
    import torch
    import torch_npu

    import tokenspeed

    root = Path(__file__).resolve().parents[3]
    for module in (tokenspeed, tokenspeed_kernel, tokenspeed_kernel_npu):
        origin = Path(module.__file__).resolve()
        if not origin.is_relative_to(root):
            raise ValueError(
                f"{module.__name__} loaded outside this checkout: {origin}"
            )

    chips = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
    if len(chips) != tp or len(set(chips)) != tp or not all(c.isdigit() for c in chips):
        raise ValueError(
            "ASCEND_RT_VISIBLE_DEVICES must list exactly TP distinct chips"
        )
    if torch.npu.device_count() != tp:
        raise ValueError("Visible NPU count differs from TP")
    config = json.loads((Path(model) / "config.json").read_text())
    metadata = dict(
        python=platform.python_version(),
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        cann_path=os.environ.get("ASCEND_HOME_PATH"),
        transformers=importlib.metadata.version("transformers"),
        chips=chips,
        model=str(Path(model).resolve()),
        tp=tp,
        geometry=model_geometry(config, tp=tp),
        checkpoint_sha256=checkpoint_identity(model),
    )
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata


def run(args):
    from transformers import AutoTokenizer

    from tokenspeed.runtime.entrypoints.engine import Engine

    metadata = inspect(args.model, tp=args.tp)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    sources = [
        tokenizer.encode(text * 800, add_special_tokens=False)
        for text in ("春天来了，校园里开满了鲜花。", "请解释计算机如何进行加法运算。")
    ]
    inputs = [source[:length] for length in (128, 2048) for source in sources]
    if any(
        len(ids) != length
        for ids, length in zip(inputs, (128, 128, 2048, 2048), strict=True)
    ):
        raise ValueError("Tokenized test source is too short")
    options = engine_options(
        args.model, tp=args.tp, enable_prefix_caching=args.prefix_cache == "enabled"
    )
    options["max_total_tokens"] = args.max_total_tokens
    sampling = dict(
        temperature=0, max_new_tokens=32, ignore_eos=True, skip_special_tokens=False
    )
    settings = {
        k: v
        for k, v in options.items()
        if k
        not in (
            "model",
            "world_size",
            "attn_tp_size",
            "dense_tp_size",
            "lm_head_tp_size",
        )
    }
    settings["concurrent_submission"] = "paused_until_queued"
    settings["output_projection_layout"] = "full_k_output_sharded"
    record = dict(
        **metadata, settings=settings, sampling=sampling, rows=[], completed=False
    )
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    batch_directory = destination.with_suffix(".batches")
    batch_directory.mkdir(exist_ok=False)
    os.environ["TOKENSPEED_TP_BATCH_CAPTURE"] = str(batch_directory.resolve())
    from batch_capture import install

    install(batch_directory)
    engine = Engine(**options)
    try:
        for phase, indices in [
            ("isolated", [0]),
            ("isolated", [1]),
            ("isolated", [2]),
            ("isolated", [3]),
            ("concurrent", [0, 3]),
            ("concurrent", [2, 1]),
            ("reuse", [0]),
            ("reuse", [3]),
        ]:
            batch = [inputs[i] for i in indices]
            result = (
                engine.llm.run(generate_queued(engine, batch, sampling))
                if phase == "concurrent"
                else engine.generate(input_ids=batch, sampling_params=sampling)
            )
            if isinstance(result, dict):
                result = [result]
            if len(result) != len(indices):
                raise ValueError("Engine returned an incomplete batch")
            for index, output in zip(indices, result, strict=True):
                record["rows"].append(
                    dict(
                        case=index,
                        phase=phase,
                        input_ids=inputs[index],
                        output_ids=output["output_ids"],
                    )
                )
            destination.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        record["batch_evidence"] = read_batch_evidence(batch_directory, tp=args.tp)
        destination.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        require_paired_decode(record)
        record["completed"] = True
        destination.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        comparison = compare_records(record, record)
        if not comparison["passed"]:
            raise AssertionError(f"Request isolation failed: {comparison}")
    finally:
        engine.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("inspect", "run"):
        command = sub.add_parser(mode)
        command.add_argument("--model", required=True)
        command.add_argument("--tp", required=True, type=int, choices=(1, 2))
        if mode == "run":
            command.add_argument(
                "--prefix-cache", required=True, choices=("enabled", "disabled")
            )
            command.add_argument("--max-total-tokens", type=int, default=16384)
            command.add_argument("--output", required=True)
    command = sub.add_parser("compare")
    command.add_argument("actual")
    command.add_argument("reference")
    args = parser.parse_args()
    if args.mode == "inspect":
        inspect(args.model, tp=args.tp)
    elif args.mode == "run":
        run(args)
    else:
        result = compare_records(
            json.loads(Path(args.actual).read_text()),
            json.loads(Path(args.reference).read_text()),
        )
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()

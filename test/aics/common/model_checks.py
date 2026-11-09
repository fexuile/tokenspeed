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

"""Shared in-memory model references and HTTP checks; no result files."""

import json
import time
import urllib.request
from pathlib import Path


def default_model():
    return str(Path.home() / "models" / "Qwen3.5-0.8B")


def prompts():
    return [
        "计算 1+1，只回答结果。",
        "你好，请用一句话介绍你自己。",
        "春天来了，校园里开满了鲜花。" * 12 + "请用一句话概括上述内容。",
        "春天来了，校园里开满了鲜花。" * 128 + "请用一句话概括上述内容。",
        "计算 1+1，只回答结果。",
    ]


def prepare_cases(model):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    cases = []
    for index, prompt in enumerate(prompts()):
        rendered = tokenizer.apply_chat_template(
            [dict(role="user", content=prompt)],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        cases.append(
            dict(
                index=index,
                rendered=rendered,
                input_ids=tokenizer.encode(rendered, add_special_tokens=False),
            )
        )
    return tokenizer, cases


def reference_results(model, tokenizer, cases):
    import torch
    import torch_npu
    from transformers import Qwen3_5ForConditionalGeneration

    torch.set_num_threads(8)
    torch.manual_seed(0)
    reference = (
        Qwen3_5ForConditionalGeneration.from_pretrained(
            model,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            local_files_only=True,
        )
        .to("npu")
        .eval()
    )
    print(f"Reference device: {next(reference.parameters()).device}", flush=True)
    results = []
    for case in cases:
        started = time.perf_counter()
        inputs = torch.tensor([case["input_ids"]], device="npu")
        with torch.inference_mode():
            generated = reference.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                do_sample=False,
                max_new_tokens=32,
                use_cache=True,
                eos_token_id=list(dict.fromkeys([tokenizer.eos_token_id, 248044])),
                pad_token_id=tokenizer.pad_token_id,
            )
        torch.npu.synchronize()
        ids = generated[0, inputs.shape[1] :].cpu().tolist()
        row = dict(
            index=case["index"],
            input_ids=case["input_ids"],
            output_tokens=len(ids),
            text=tokenizer.decode(ids, skip_special_tokens=True),
            elapsed_s=time.perf_counter() - started,
        )
        results.append(row)
        print_case("REFERENCE", row)
    del reference
    torch.npu.empty_cache()
    return results


def server_result(case, base_url, served_model):
    started = time.perf_counter()
    body = dict(
        model=served_model,
        prompt=case["rendered"],
        max_tokens=32,
        temperature=0,
        stream=False,
    )
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = json.load(response)
    assert payload["usage"]["prompt_tokens"] == len(case["input_ids"]), payload
    return dict(
        index=case["index"],
        input_ids=case["input_ids"],
        output_tokens=payload["usage"]["completion_tokens"],
        text=payload["choices"][0]["text"],
        elapsed_s=time.perf_counter() - started,
    )


def print_case(label, row):
    print(
        f"{label} case={row['index']} input={len(row['input_ids'])} "
        f"output={row['output_tokens']} elapsed={row['elapsed_s']:.3f}s "
        f"text={row['text']!r}",
        flush=True,
    )


def compare_results(actual, expected):
    if not actual or len(actual) != len(expected):
        raise ValueError(
            "Both result lists must contain the same nonzero number of cases"
        )
    comparisons = []
    for a, b in zip(actual, expected):
        comparisons.append(
            dict(
                index=a["index"],
                same_input_ids=a["input_ids"] == b["input_ids"],
                exact_text=a["text"] == b["text"],
            )
        )
    repeat = actual[0]["text"] == actual[-1]["text"]
    passed = repeat and all(
        r["same_input_ids"] and r["exact_text"] for r in comparisons
    )
    return dict(
        comparisons=comparisons, repeated_request_identical=repeat, passed=passed
    )

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

"""Behavioral tests for persisted TP acceptance records and launch validation."""

import copy
import subprocess
import sys
from pathlib import Path

import pytest
from tp_validation import (
    checkpoint_identity,
    compare_records,
    model_geometry,
    read_batch_evidence,
)


def record():
    return dict(
        completed=True,
        tp=1,
        batch_evidence={"0": {"paired_decode_steps": 62, "max_batch_size": 2}},
        checkpoint_sha256="same",
        settings={"dtype": "bfloat16"},
        sampling={"max_new_tokens": 32},
        rows=[
            dict(case=0, phase=phase, input_ids=[1, 2], output_ids=[3] * 32)
            for phase in ("isolated", "concurrent", "reuse")
        ],
    )


def test_comparison_and_isolation():
    expected = record()
    assert compare_records(expected, copy.deepcopy(expected))["passed"]
    actual = copy.deepcopy(expected)
    actual["rows"][-1]["output_ids"][0] = 4
    assert not compare_records(actual, expected)["passed"]
    assert not compare_records(actual, actual)["passed"]


@pytest.mark.parametrize("key", ["checkpoint_sha256", "settings", "sampling"])
def test_incompatible_runs_are_rejected(key):
    actual = record()
    actual[key] = "different"
    with pytest.raises(ValueError):
        compare_records(actual, record())


def test_missing_decode_tokens_are_rejected():
    actual = record()
    actual["rows"][0]["output_ids"].pop()
    with pytest.raises(ValueError):
        compare_records(actual, record())


def test_checkpoint_identity_includes_weight_contents(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"first")
    first = checkpoint_identity(tmp_path)
    weight.write_bytes(b"other")
    assert checkpoint_identity(tmp_path) != first


def test_geometry_splits_each_gdn_component():
    config = dict(
        hidden_size=1024,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=256,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        intermediate_size=3584,
        vocab_size=248320,
    )
    one = model_geometry(config, tp=1)
    two = model_geometry(config, tp=2)
    assert two["qkv_weight"] == [2560, 1024]
    assert two["attention_output_weight"] == [512, 2048]
    assert two["gdn_output_weight"] == [512, 2048]
    assert two["down_weight"] == [512, 3584]
    assert two["convolution_state"] == [3072, 3]
    assert two["recurrent_state"] == [8, 128, 128]
    assert two["gdn_projection_weight"][0] * 2 == one["gdn_projection_weight"][0]


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["model", "2", "6,7"],
        ["model", "2", "6,7", "invalid"],
        ["model", "3", "6,7", "disabled"],
        ["model", "2", "6", "enabled"],
        ["model", "2", "6,6", "enabled"],
    ],
)
def test_launch_rejects_invalid_chip_selection(args):
    result = subprocess.run(
        ["bash", str(Path(__file__).with_name("serve-qwen35.sh")), *args],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2


def test_partial_run_is_rejected():
    actual = record()
    actual["completed"] = False
    with pytest.raises(ValueError):
        compare_records(actual, actual)


def test_run_requires_explicit_cache_policy(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("tp_validation.py")),
            "run",
            "--model",
            "unused",
            "--tp",
            "2",
            "--output",
            str(tmp_path / "result.json"),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "--prefix-cache" in result.stderr
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("paired", [0, 1, 32, 59])
def test_serialized_concurrent_requests_are_rejected(paired):
    actual = record()
    actual["batch_evidence"]["0"]["paired_decode_steps"] = paired
    with pytest.raises(ValueError, match="two-request decode evidence"):
        compare_records(actual, actual)


def test_missing_rank_batch_evidence_is_rejected():
    actual = record()
    actual["tp"] = 2
    with pytest.raises(ValueError, match="Rank 1"):
        compare_records(actual, actual)


def test_batch_evidence_counts_decode_not_prefill(tmp_path):
    import json

    events = (
        [dict(bs=2, num_extends=2, tokens=2176)] * 5
        + [dict(bs=2, num_extends=0, tokens=2)] * 62
        + [dict(bs=1, num_extends=0, tokens=1)] * 32
    )
    (tmp_path / "rank0.jsonl").write_text("".join(f"{json.dumps(e)}\n" for e in events))
    assert read_batch_evidence(tmp_path, tp=1) == {
        "0": dict(paired_decode_steps=62, max_batch_size=2)
    }

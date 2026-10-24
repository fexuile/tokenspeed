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

"""Behavioral checks for the in-memory comparison and latency helpers."""

import copy

import pytest
from bench_engine import measure_stream, summarize_runs
from model_checks import compare_results


def cases():
    return [
        dict(index=0, input_ids=[1, 2], text="2"),
        dict(index=1, input_ids=[3, 4], text="hello"),
        dict(index=2, input_ids=[1, 2], text="2"),
    ]


def test_matching_results_pass():
    result = compare_results(cases(), copy.deepcopy(cases()))
    assert result["passed"] and result["repeated_request_identical"]


@pytest.mark.parametrize("field,value", [("text", "wrong"), ("input_ids", [99])])
def test_reference_difference_fails(field, value):
    actual = cases()
    actual[1][field] = value
    assert not compare_results(actual, cases())["passed"]


def test_repeat_is_checked_even_when_reference_texts_match():
    actual = cases()
    actual[-1]["text"] = "changed"
    result = compare_results(actual, copy.deepcopy(actual))
    assert all(row["exact_text"] for row in result["comparisons"])
    assert not result["passed"] and not result["repeated_request_identical"]


@pytest.mark.parametrize("actual,expected", [([], []), (cases(), cases()[:1])])
def test_missing_cases_are_rejected(actual, expected):
    with pytest.raises(ValueError):
        compare_results(actual, expected)


def test_stream_counts_each_token_once_and_excludes_prefill_from_tpot():
    stream = [
        dict(meta_info=dict(completion_tokens=n, prompt_tokens=128)) for n in (1, 2, 2)
    ]
    times = iter([10.1, 10.14, 10.2])
    row = measure_stream(
        iter(stream), expected_output=2, started=10, clock=lambda: next(times)
    )
    assert row["input_tokens"] == 128 and row["output_tokens"] == 2
    assert row["ttft_ms"] == pytest.approx(100)
    assert row["decode_ms_per_token"] == pytest.approx(40)
    assert row["e2e_ms"] == pytest.approx(140)


def test_missing_streaming_token_is_rejected():
    stream = [
        dict(meta_info=dict(completion_tokens=n, prompt_tokens=128)) for n in (1, 3)
    ]
    with pytest.raises(AssertionError, match="one streaming arrival"):
        measure_stream(iter(stream), expected_output=3, started=0, clock=lambda: 1)


def test_short_generation_is_rejected():
    stream = [dict(meta_info=dict(completion_tokens=1, prompt_tokens=128))]
    with pytest.raises(AssertionError):
        measure_stream(iter(stream), expected_output=2, started=0, clock=lambda: 1)


def test_summary_uses_sample_standard_deviation():
    rows = [
        dict(ttft_ms=100, decode_ms_per_token=40, e2e_ms=140),
        dict(ttft_ms=120, decode_ms_per_token=50, e2e_ms=170),
    ]
    result = summarize_runs(rows)
    assert result["ttft_ms"]["mean"] == 110
    assert result["ttft_ms"]["stdev"] == pytest.approx(2**0.5 * 10)
    assert result["decode_ms_per_token"]["median"] == 45


def test_single_sample_summary_has_zero_spread():
    result = summarize_runs([dict(ttft_ms=100, decode_ms_per_token=40, e2e_ms=140)])
    assert result["ttft_ms"]["stdev"] == 0


def test_empty_summary_is_rejected():
    with pytest.raises(ValueError):
        summarize_runs([])

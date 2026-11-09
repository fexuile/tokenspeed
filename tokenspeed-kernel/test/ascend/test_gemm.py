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

import pytest
import torch

pytest.importorskip("torch_npu")

from tokenspeed_kernel.ops.gemm import bmm, mm, plan_dense_gemm
from tokenspeed_kernel_npu.ops.gemm import _ascend_mm_kernel
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend GEMM tests require an NPU"
)


@pytest.mark.parametrize("hidden,output", [(2560, 12352), (5120, 8240)])
def test_mm_does_not_depend_on_other_rows(hidden, output):
    torch.manual_seed(83)
    x = torch.randn(2176, hidden, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(output, hidden, device="npu", dtype=torch.bfloat16)
    weight = weight / hidden**0.5
    packed = mm(x, weight)
    for start in (0, 7, 2048):
        for rows in (1, 2, 17, 128):
            single = mm(x[start : start + rows], weight)
            torch.testing.assert_close(
                single, packed[start : start + rows], atol=0, rtol=0
            )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dimensions", [(35, 257, 129), (3, 128, 128)])
@pytest.mark.parametrize("use_bias", [False, True])
def test_mm_matches_fp32_projection(dtype, dimensions, use_bias):
    rows, columns, inner = dimensions
    torch.manual_seed(63)
    # Exercise row strides in the inputs, weights and supplied output.
    x = torch.randn(rows, inner + 7, device="npu", dtype=dtype)[:, :inner]
    weight = torch.randn(columns, inner + 3, device="npu", dtype=dtype)[:, :inner]
    weight = weight / inner**0.5
    bias = torch.randn(columns, device="npu", dtype=dtype) if use_bias else None
    storage = torch.full((rows, columns + 5), 19.0, device="npu", dtype=dtype)
    out = storage[:, :columns]
    expected = torch.nn.functional.linear(
        x.cpu().float(),
        weight.cpu().float(),
        None if bias is None else bias.cpu().float(),
    )

    actual = mm(x, weight, bias=bias, out=out)

    assert actual is out
    torch.testing.assert_close(
        actual.cpu().float(), expected, atol=2e-5, rtol=torch.finfo(dtype).eps / 2
    )
    torch.testing.assert_close(
        storage[:, columns:].cpu(),
        torch.full((rows, 5), 19.0, dtype=dtype),
        atol=0,
        rtol=0,
    )


def test_mm_epilogue_preserves_alpha_and_bias():
    torch.manual_seed(29)
    x = torch.randn(7, 129, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(65, 129, device="npu", dtype=torch.bfloat16)
    bias = torch.randn(65, device="npu", dtype=torch.bfloat16)
    alpha = torch.tensor(0.25, device="npu")
    base = mm(x, weight)
    expected = (base * alpha.to(base.dtype) + bias).float()
    actual = mm(x, weight, alpha=alpha, bias=bias, out_dtype=torch.float32)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_mm_reuses_compilation_across_row_counts():
    torch.manual_seed(91)
    x = torch.randn(2176, 256, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(384, 256, device="npu", dtype=torch.bfloat16)
    mm(x[:128], weight)
    with assert_no_triton_compile(_ascend_mm_kernel):
        for rows in (1, 7, 33, 129, 2176):
            mm(x[:rows], weight)


def test_cpu_projection_on_an_ascend_host():
    x = torch.arange(15, dtype=torch.bfloat16).view(3, 5)
    weight = torch.arange(35, dtype=torch.bfloat16).view(7, 5)
    torch.testing.assert_close(
        mm(x, weight), torch.nn.functional.linear(x, weight), atol=0, rtol=0
    )


def test_mm_covers_more_than_65535_tiles():
    # 128 row tiles * 544 column tiles exceeds Ascend's launch grid limit.
    x = torch.ones((2048, 32), device="npu", dtype=torch.bfloat16)
    columns = torch.arange(34816, device="npu") % 5
    weight = (columns.to(torch.bfloat16)[:, None].expand(-1, 32) / 32).contiguous()
    actual = mm(x, weight)
    expected = columns.to(torch.bfloat16)[None, :].expand(2048, -1)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["mm", "bmm"])
def test_cpu_dense_plan_on_an_ascend_host(mode):
    with torch.device("cpu"):
        plan = plan_dense_gemm(
            mode=mode,
            input_size=5,
            output_size=7,
            dtype=torch.bfloat16,
            batch_size=2 if mode == "bmm" else None,
            batch_invariant=False,
        )
        if mode == "mm":
            x = torch.arange(15, dtype=torch.bfloat16).view(3, 5)
            weight = torch.arange(35, dtype=torch.bfloat16).view(7, 5)
            actual = mm(x, weight, plan=plan)
            expected = torch.nn.functional.linear(x, weight)
        else:
            x = torch.arange(30, dtype=torch.bfloat16).view(2, 3, 5)
            weight = torch.arange(70, dtype=torch.bfloat16).view(2, 7, 5)
            actual = bmm(x, weight, plan=plan)
            expected = torch.bmm(x, weight.transpose(1, 2))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

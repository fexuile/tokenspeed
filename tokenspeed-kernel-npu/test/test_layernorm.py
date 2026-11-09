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

torch_npu = pytest.importorskip("torch_npu")

from tokenspeed_kernel.ops.layernorm import qk_rmsnorm
from tokenspeed_kernel_npu.ops.layernorm import rmsnorm

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend RMSNorm tests require an NPU"
)


def _reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_float = x.float()
    variance = x_float.square().mean(dim=-1, keepdim=True)
    return (x_float * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rmsnorm(dtype: torch.dtype) -> None:
    x = torch.randn(7, 1024, device="npu", dtype=dtype)
    weight = torch.randn(1024, device="npu", dtype=dtype)
    expected = _reference(x, weight, 1e-6)

    out = torch.empty_like(x)
    result = rmsnorm(x, weight, 1e-6, out=out)

    assert result is out
    torch.testing.assert_close(result, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_add_rmsnorm_updates_residual(dtype: torch.dtype) -> None:
    x = torch.randn(7, 1024, device="npu", dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.randn(1024, device="npu", dtype=dtype)
    expected_residual = x + residual
    expected = _reference(expected_residual, weight, 1e-6)

    result, residual_out = rmsnorm(x, weight, 1e-6, residual=residual)

    assert residual_out is residual
    torch.testing.assert_close(residual, expected_residual, atol=0, rtol=0)
    torch.testing.assert_close(result, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("weight_offset", [0.0, 1.0])
@pytest.mark.parametrize("head_shape", [(7, 4 * 256), (7, 4, 256)])
def test_qk_rmsnorm_keeps_affine_in_fp32(dtype, weight_offset, head_shape):
    # Packed Q/K views must retain their rows while every head normalizes alone.
    generator = torch.Generator().manual_seed(57)
    packed = torch.randn(7, 8 * 256, generator=generator, dtype=dtype).to("npu")
    q = packed[:, : 4 * 256].view(head_shape)
    k = packed[:, 4 * 256 : 6 * 256]
    weights = [
        (torch.randn(256, generator=generator) * 0.2).to(dtype).to("npu")
        for _ in range(2)
    ]
    originals = [x.cpu().clone() for x in (q, k)]
    expected = [
        _reference(x.reshape(-1, 256), w.float() + weight_offset, 1e-6)
        .reshape(x.shape)
        .cpu()
        for x, w in zip((q, k), weights, strict=True)
    ]

    outputs = qk_rmsnorm(q, k, *weights, 1e-6, weight_offset=weight_offset)

    for actual, reference, original, source in zip(
        outputs, expected, originals, (q, k), strict=True
    ):
        assert actual.shape == source.shape and actual.dtype == dtype
        torch.testing.assert_close(actual.cpu(), reference, atol=0, rtol=0)
        torch.testing.assert_close(source.cpu(), original, atol=0, rtol=0)


@pytest.mark.parametrize("q_heads,k_heads", [(4, 2), (8, 4)])
def test_qk_rmsnorm_does_not_depend_on_other_rows(q_heads, k_heads):
    generator = torch.Generator().manual_seed(97)
    q = torch.randn(128, q_heads * 256, generator=generator, dtype=torch.bfloat16).to(
        "npu"
    )
    k = torch.randn(128, k_heads * 256, generator=generator, dtype=torch.bfloat16).to(
        "npu"
    )
    weights = [
        (torch.randn(256, generator=generator) * 0.1).to(device="npu", dtype=q.dtype)
        for _ in range(2)
    ]
    packed = qk_rmsnorm(q, k, *weights, 1e-6, weight_offset=1.0)
    for start in (0, 7, 119):
        for rows in (1, 2, 7):
            alone = qk_rmsnorm(
                q[start : start + rows],
                k[start : start + rows],
                *weights,
                1e-6,
                weight_offset=1.0
            )
            for actual, reference in zip(alone, packed, strict=True):
                torch.testing.assert_close(
                    actual, reference[start : start + rows], atol=0, rtol=0
                )

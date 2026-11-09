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

from tokenspeed.runtime.layers.layernorm import GemmaRMSNorm

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend norms require an NPU"
)


@pytest.mark.parametrize("width", [1024, 2560, 4096, 5120])
def test_gemma_norm_does_not_depend_on_other_rows(width):
    generator = torch.Generator().manual_seed(127)
    x = torch.randn(128, width, generator=generator, dtype=torch.bfloat16).to("npu")
    norm = GemmaRMSNorm(width).to(device="npu", dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_((torch.randn(width, generator=generator) * 0.1).to("npu"))
    packed = norm(x)
    for row in range(128):
        alone = norm(x[row : row + 1])
        torch.testing.assert_close(alone, packed[row : row + 1], atol=0, rtol=0)
        paired = norm(torch.cat((x[:1], x[row : row + 1])))[-1:]
        torch.testing.assert_close(alone, paired, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("with_residual", [False, True])
def test_gemma_norm_matches_fp32_formula(dtype, with_residual):
    generator = torch.Generator().manual_seed(71)
    # Non-power-of-two widths exercise masked reduction lanes.
    x_cpu = torch.randn(7, 257, generator=generator, dtype=dtype)
    residual_cpu = torch.randn(7, 257, generator=generator, dtype=dtype)
    norm = GemmaRMSNorm(257, eps=1e-5).to(device="npu", dtype=dtype)
    weight = (torch.randn(257, generator=generator) * 0.1).to(dtype)
    with torch.no_grad():
        norm.weight.copy_(weight.to("npu"))
    x = x_cpu.to("npu")
    residual = residual_cpu.to("npu") if with_residual else None
    summed = x_cpu + residual_cpu if with_residual else x_cpu
    reference = summed.float()
    reference = reference * torch.rsqrt(
        reference.square().mean(-1, keepdim=True) + 1e-5
    )
    reference = reference * (1 + weight.float())

    result = norm(x, residual)

    if with_residual:
        result, returned_residual = result
        torch.testing.assert_close(returned_residual.cpu(), summed, atol=0, rtol=0)
        torch.testing.assert_close(residual.cpu(), residual_cpu, atol=0, rtol=0)
    torch.testing.assert_close(x.cpu(), x_cpu, atol=0, rtol=0)
    # FP32 reduction trees can straddle an output-cast midpoint. Compare before
    # reference rounding, within half the activation dtype's relative spacing.
    torch.testing.assert_close(
        result.cpu().float(), reference, atol=2e-6, rtol=torch.finfo(dtype).eps / 2
    )


def test_gemma_norm_accepts_cpu_tensors_on_ascend_host():
    x = torch.arange(21, dtype=torch.bfloat16).view(3, 7)
    norm = GemmaRMSNorm(7).to(dtype=x.dtype)
    reference = x.float() * torch.rsqrt(
        x.float().square().mean(-1, keepdim=True) + 1e-6
    )
    torch.testing.assert_close(norm(x), reference.to(x.dtype), atol=0, rtol=0)

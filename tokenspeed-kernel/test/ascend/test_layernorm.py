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

from tokenspeed_kernel.ops.layernorm import grouped_gemma_rmsnorm
from tokenspeed_kernel.ops.layernorm.triton import _grouped_gemma_rmsnorm_kernel
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend normalization requires an NPU"
)


def test_gemma_norm_reuses_compilation_across_row_counts():
    torch.manual_seed(67)
    x = torch.randn(2048, 2560, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(2560, device="npu", dtype=torch.bfloat16)
    grouped_gemma_rmsnorm(x[:128], weight, 2560, 1e-6, out=None)
    with assert_no_triton_compile(_grouped_gemma_rmsnorm_kernel):
        for rows in (1, 2, 17, 129, 2048):
            grouped_gemma_rmsnorm(x[:rows], weight, 2560, 1e-6, out=None)


def test_grouped_gemma_norm_preserves_output_buffer():
    torch.manual_seed(19)
    x = torch.randn(17, 512, device="npu", dtype=torch.bfloat16)
    weight = torch.randn(512, device="npu", dtype=torch.bfloat16)
    out = torch.empty_like(x)
    result = grouped_gemma_rmsnorm(x, weight, 256, 1e-6, out=out)
    assert result is out
    for group in range(2):
        start = group * 256
        separate = grouped_gemma_rmsnorm(
            x[:, start : start + 256], weight[start : start + 256], 256, 1e-6, out=None
        )
        torch.testing.assert_close(
            result[:, start : start + 256], separate, atol=0, rtol=0
        )

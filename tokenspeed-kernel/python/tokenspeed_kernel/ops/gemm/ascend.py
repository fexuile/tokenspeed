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

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

if current_platform().is_npu:
    from tokenspeed_kernel_npu.ops.gemm import launch_ascend_mm

    @register_kernel(
        "gemm",
        "mm",
        name="ascend_triton_mm",
        solution="ascend",
        capability=CapabilityRequirement(vendors=frozenset({"ascend"})),
        signatures=format_signatures(
            ("a", "b"), "dense", {torch.bfloat16, torch.float16}
        ),
        features={"batch_invariant"},
        traits={
            "device_type": frozenset({"npu"}),
            "a_inner_stride_one": frozenset({True}),
            "b_inner_stride_one": frozenset({True}),
            "out_inner_stride_one": frozenset({True}),
            "out_dtype": frozenset({torch.bfloat16, torch.float16, torch.float32}),
        },
        priority=Priority.PERFORMANT,
    )
    def ascend_triton_mm(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scales: torch.Tensor | None,
        B_scales: torch.Tensor | None,
        out_dtype: torch.dtype,
        *,
        alpha: torch.Tensor | None = None,
        block_size: list[int] | None = None,
        bias: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dense Ascend GEMM; retain Torch's rounded alpha/bias epilogue order."""
        if A_scales is not None or B_scales is not None or block_size is not None:
            raise ValueError("Dense Ascend GEMM does not accept quantization scales")
        if out is not None and out.dtype != A.dtype:
            raise ValueError("Ascend GEMM out must have the input dtype")
        result = launch_ascend_mm(A, B, bias=bias if alpha is None else None, out=out)
        if alpha is not None:
            result.mul_(alpha.to(dtype=result.dtype))
            if bias is not None:
                result.add_(bias.to(dtype=result.dtype))
        return result.to(out_dtype)

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

"""Ascend normalization operators."""

from __future__ import annotations

import torch
import torch_npu


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply RMSNorm, optionally adding and updating a residual tensor."""
    if x.shape[0] == 0:
        result = x if out is None else out
        return (result, residual) if residual is not None else result
    if residual is not None:
        result, _, residual_out = torch_npu.npu_add_rms_norm(
            x, residual, weight, epsilon=eps
        )
        residual.copy_(residual_out)
        return result, residual
    result, _ = torch_npu.npu_rms_norm(x, weight, epsilon=eps)
    if out is not None:
        out.copy_(result)
        result = out
    return result


def qk_rmsnorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    *,
    weight_offset: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize Q/K heads in FP32 and round each affine result once.

    Args:
        q, k: Packed or head-shaped query/key rows with dense head channels.
        q_weight, k_weight: Raw per-head affine weights.
        eps: Epsilon added to each head's mean square.
        weight_offset: Offset added in FP32 (1.0 for Gemma-style weights).

    Returns:
        Normalized Q/K with their original shapes and dtypes.
    """
    head_dim = q_weight.numel()
    if k_weight.numel() != head_dim:
        raise ValueError("q and k weights must have the same head dimension")
    if q.shape[-1] % head_dim or k.shape[-1] % head_dim:
        raise ValueError("q and k widths must be divisible by the head dimension")

    q_shape, k_shape = q.shape, k.shape
    q_heads = q.reshape(-1, q.shape[-1] // head_dim, head_dim)
    k_heads = k.reshape(-1, k.shape[-1] // head_dim, head_dim)
    q_float, k_float = q_heads.float(), k_heads.float()
    q_scale = torch.rsqrt(q_float.square().mean(-1, keepdim=True) + eps)
    k_scale = torch.rsqrt(k_float.square().mean(-1, keepdim=True) + eps)
    # Qwen's Gemma-style affine keeps both the offset and RMS scale in FP32.
    q_out = (q_float * q_scale * (q_weight.float() + weight_offset)).to(q.dtype)
    k_out = (k_float * k_scale * (k_weight.float() + weight_offset)).to(k.dtype)
    q_out, k_out = q_out.reshape(q_shape), k_out.reshape(k_shape)
    return q_out, k_out


__all__ = ["qk_rmsnorm", "rmsnorm"]

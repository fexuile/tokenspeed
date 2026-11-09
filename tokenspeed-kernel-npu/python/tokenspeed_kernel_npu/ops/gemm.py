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
import torch_npu
from tokenspeed_kernel_npu._triton import tl, triton


def prepare_weight_nz(weight: torch.Tensor, *, transpose: bool = False) -> torch.Tensor:
    if weight.device.type != "npu" or weight.ndim != 2:
        raise ValueError("Weight-NZ preparation requires a two-dimensional NPU tensor.")
    if transpose:
        weight = weight.transpose(0, 1).contiguous()

    config = torch.npu.config
    previous = getattr(config, "allow_internal_format", None)
    try:
        config.allow_internal_format = True
        prepared = torch_npu.npu_format_cast(weight, 29)
        actual_format = torch_npu.get_npu_format(prepared)
        if actual_format != 29:
            raise RuntimeError(
                f"Weight-NZ preparation requested format 29, got {actual_format}."
            )
        return prepared
    finally:
        config.allow_internal_format = previous


class PreparedInt8Linear(torch.nn.Module):
    """Post-load NZ weights for dynamic-token INT8 dense matmul."""

    def __init__(self, weight, weight_scale, *, output_dtype):
        super().__init__()
        if weight.dtype != torch.int8 or weight.ndim != 2:
            raise ValueError("INT8 linear requires [N,K] INT8 weights")
        if weight_scale.numel() != weight.shape[0]:
            raise ValueError("INT8 linear requires one scale per output channel")
        if output_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("INT8 linear output must be BF16 or FP16")
        self.output_dtype = output_dtype
        self.register_buffer(
            "weight", prepare_weight_nz(weight, transpose=True), persistent=False
        )
        self.register_buffer(
            "weight_scale", weight_scale.flatten().float(), persistent=False
        )

    def forward(self, x, *, bias=None, defer_dequant=False):
        if defer_dequant and bias is not None:
            raise ValueError("Deferred INT8 dequantization requires a bias-free linear")
        if isinstance(x, tuple):
            values, scales = x
            if values.dtype != torch.int8 or scales.dtype != torch.float32:
                raise ValueError(
                    "Prequantized input requires INT8 values and FP32 scales"
                )
        else:
            if x.numel() == 0:
                values = x.to(torch.int8)
                scales = x.new_empty(x.shape[:-1], dtype=torch.float32)
            else:
                values, scales = torch_npu.npu_dynamic_quant(x)
        output_shape = values.shape[:-1] + (self.weight.shape[1],)
        if values.numel() == 0:
            result = values.new_empty(
                output_shape,
                dtype=torch.int32 if defer_dequant else self.output_dtype,
            )
        else:
            result = torch_npu.npu_quant_matmul(
                values.reshape(-1, values.shape[-1]),
                self.weight,
                self.weight_scale,
                pertoken_scale=None if defer_dequant else scales.reshape(-1),
                bias=bias,
                output_dtype=torch.int32 if defer_dequant else self.output_dtype,
            ).reshape(output_shape)
        return (result, scales) if defer_dequant else result


@triton.jit(
    do_not_specialize=["rows", "input_stride", "weight_stride", "output_stride"]
)
def _ascend_mm_kernel(
    x,
    weight,
    out,
    rows,
    columns: tl.constexpr,
    inner: tl.constexpr,
    input_stride,
    weight_stride,
    output_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    column_tiles: tl.constexpr = tl.cdiv(columns, BLOCK_N)
    total_tiles = tl.cdiv(rows, BLOCK_M) * column_tiles
    # Ascend caps the launch at 65535 blocks; excess tiles stay on this grid.
    for tile in range(tl.program_id(0), total_tiles, tl.num_programs(0)):
        row = (tile // column_tiles) * BLOCK_M + tl.arange(0, BLOCK_M)
        column = (tile % column_tiles) * BLOCK_N + tl.arange(0, BLOCK_N)
        offset = tl.arange(0, BLOCK_K)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        # Fixed tiles and K traversal keep a row's reduction independent of M.
        for block in range(tl.cdiv(inner, BLOCK_K)):
            k = block * BLOCK_K + offset
            a = tl.load(
                x + row[:, None].to(tl.int64) * input_stride + k[None, :],
                mask=(row[:, None] < rows) & (k[None, :] < inner),
                other=0,
            )
            b = tl.load(
                weight + column[None, :].to(tl.int64) * weight_stride + k[:, None],
                mask=(column[None, :] < columns) & (k[:, None] < inner),
                other=0,
            )
            accumulator = tl.dot(a, b, accumulator)
        tl.store(
            out + row[:, None].to(tl.int64) * output_stride + column[None, :],
            accumulator,
            mask=(row[:, None] < rows) & (column[None, :] < columns),
        )


def launch_ascend_mm(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    bias: torch.Tensor | None,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """Compute ``x @ weight.T + bias`` with a fixed FP32 reduction order.

    Args:
        x: BF16 or FP16 NPU rows ``[M, K]`` with unit inner stride.
        weight: Same-dtype NPU weights ``[N, K]`` with unit inner stride.
        bias: Optional same-dtype contiguous ``[N]`` affine bias.
        out: Optional ``[M, N]`` same-dtype output with unit inner stride.

    Returns:
        The result in x's dtype, using out's storage when supplied.
    """
    if (
        x.device.type != "npu"
        or x.ndim != 2
        or weight.ndim != 2
        or x.device != weight.device
        or x.dtype not in (torch.bfloat16, torch.float16)
        or x.dtype != weight.dtype
        or x.shape[1] != weight.shape[1]
        or x.stride(-1) != 1
        or weight.stride(-1) != 1
    ):
        raise ValueError("Ascend GEMM requires matching BF16/FP16 row-strided matrices")
    rows, columns = x.shape[0], weight.shape[0]
    if bias is not None and (
        bias.device != x.device
        or bias.dtype != x.dtype
        or bias.shape != (columns,)
        or not bias.is_contiguous()
    ):
        raise ValueError("Ascend GEMM bias must match its output columns and dtype")
    if out is None:
        out = torch.empty((rows, columns), device=x.device, dtype=x.dtype)
    elif (
        out.shape != (rows, columns)
        or out.device != x.device
        or out.dtype != x.dtype
        or out.stride(-1) != 1
    ):
        raise ValueError(
            "Ascend GEMM out must match the result and have unit inner stride"
        )
    if rows == 0 or columns == 0:
        return out
    accumulator = (
        torch.empty((rows, columns), device=x.device, dtype=torch.float32)
        if bias is not None
        else out
    )
    tiles = triton.cdiv(rows, 16) * triton.cdiv(columns, 64)
    _ascend_mm_kernel[(min(tiles, 65535),)](
        x,
        weight,
        accumulator,
        rows,
        columns,
        x.shape[1],
        x.stride(0),
        weight.stride(0),
        accumulator.stride(0),
        BLOCK_M=16,
        BLOCK_N=64,
        BLOCK_K=128,
    )
    if bias is not None:
        # Materialize the FP32 accumulator for the affine epilogue. Triton-Ascend
        # can miscompile a broadcast bias following a single Cube K tile.
        out.copy_(accumulator + bias.float())
    return out

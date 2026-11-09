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

from tokenspeed_kernel.ops.attention.mha import (
    mha_decode_with_kvcache,
    mha_extend_with_kvcache,
)

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend attention requires an NPU"
)


@pytest.mark.parametrize("heads,kv_heads", [(24, 4), (32, 4)])
@pytest.mark.parametrize("length", [128, 129, 2048])
@pytest.mark.parametrize("decode", [False, True])
def test_paged_attention_preserves_head_partition(heads, kv_heads, length, decode):
    torch.manual_seed(94)
    dtype = torch.bfloat16
    q = torch.randn(length, heads, 256, device="npu", dtype=dtype)
    pages = (length + 127) // 128
    k, v = [
        torch.randn(pages, 128, kv_heads, 256, device="npu", dtype=dtype)
        for _ in range(2)
    ]
    cu = torch.tensor([0, length], device="npu", dtype=torch.int32)
    lengths = torch.tensor([length], device="npu", dtype=torch.int32)
    table = torch.arange(pages, device="npu", dtype=torch.int32)[None]

    def run(query, keys, values):
        if decode:
            return mha_decode_with_kvcache(
                query[-1:],
                keys,
                values,
                table,
                lengths,
                max_seqlen_q=1,
                max_seqlen_k=length,
            )
        return mha_extend_with_kvcache(
            query,
            cu,
            cu,
            keys,
            values,
            table,
            lengths,
            max_seqlen_q=length,
            max_seqlen_k=length,
            is_causal=True,
        )

    full = run(q, k, v)
    parts = []
    for rank in range(2):
        query_slice = slice(rank * heads // 2, (rank + 1) * heads // 2)
        kv_slice = slice(rank * kv_heads // 2, (rank + 1) * kv_heads // 2)
        parts.append(
            run(
                q[:, query_slice].contiguous(),
                k[:, :, kv_slice].contiguous(),
                v[:, :, kv_slice].contiguous(),
            )
        )
    torch.testing.assert_close(full, torch.cat(parts, dim=1), atol=0, rtol=0)

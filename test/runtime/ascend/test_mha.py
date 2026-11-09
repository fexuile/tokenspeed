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

from test.runtime.cache_pool_test_utils import make_mha_memory_plan, make_pool

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.paged.mha import MHAAttnBackend
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.kv_cache.mha import MHATokenToKVPool
from tokenspeed.runtime.layers.paged_attention import PagedAttention

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend attention requires an NPU"
)


@pytest.mark.parametrize("head_dim", [64, 256])
@pytest.mark.parametrize("neighbor_query", [128, 1536, 2048])
@pytest.mark.parametrize("heads,kv_heads", [(8, 2), (32, 4), (24, 4), (12, 2)])
def test_fresh_request_does_not_depend_on_neighbors_prefix(
    head_dim, neighbor_query, heads, kv_heads
):
    torch.manual_seed(83)
    dtype = torch.bfloat16
    spec = MHAConfig(
        backend_name="mha",
        num_attention_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        attn_tp_size=1,
    )
    config = AttnConfig(
        device="npu",
        dtype=dtype,
        kv_cache_dtype=dtype,
        kv_cache_quant_method="none",
        prefix_granularity=128,
        context_len=4096,
        max_bs=2,
        components=(spec,),
    )
    backend = MHAAttnBackend(config, spec, kernel_page_size=128)
    plan = make_mha_memory_plan(
        size=4096,
        prefix_granularity=128,
        layer_num=1,
        kv_heads=kv_heads,
        head_dim=head_dim,
        dtype=dtype,
    )
    _, pool = make_pool(
        MHATokenToKVPool,
        plan,
        device="npu",
        dtype=dtype,
        head_num=kv_heads,
        head_dim=head_dim,
        layer_num=1,
        rank=0,
    )
    layer = PagedAttention(
        heads, head_dim, head_dim**-0.5, kv_heads, 0, rotary_emb=None, qk_norm=None
    )
    prefix = 512
    neighbor_length = prefix + neighbor_query
    q = torch.randn(neighbor_query + 128, heads, head_dim, device="npu", dtype=dtype)
    k = torch.randn(
        neighbor_length + 128, kv_heads, head_dim, device="npu", dtype=dtype
    )
    v = torch.randn_like(k)
    # The neighbor and fresh request occupy distinct physical cache pages.
    pool.get_key_buffer(0)[128 : 128 + len(k)].copy_(k)
    pool.get_value_buffer(0)[128 : 128 + len(v)].copy_(v)
    pages = neighbor_length // 128
    table_cpu = torch.zeros((2, 32), dtype=torch.int32)
    table_cpu[0, :pages] = torch.arange(1, pages + 1, dtype=torch.int32)
    table_cpu[1, 0] = pages + 1
    table = table_cpu.to("npu")
    cache_locs = torch.arange(
        128 + prefix, 128 + len(k), device="npu", dtype=torch.int64
    )

    def run(*, packed):
        lengths_cpu = torch.tensor(
            [neighbor_length, 128] if packed else [128], dtype=torch.int32
        )
        prefixes_cpu = torch.tensor([prefix, 0] if packed else [0], dtype=torch.int32)
        queries_cpu = torch.tensor(
            [neighbor_query, 128] if packed else [128], dtype=torch.int32
        )
        batch = len(lengths_cpu)
        backend.init_forward_metadata(
            batch,
            batch,
            lengths_cpu.to("npu"),
            table if packed else table[1:],
            ForwardMode.EXTEND,
            extend_seq_lens=queries_cpu.to("npu"),
            extend_seq_lens_cpu=queries_cpu,
            extend_prefix_lens=prefixes_cpu.to("npu"),
            extend_prefix_lens_cpu=prefixes_cpu,
            extend_with_prefix=packed,
            query_shard=None,
            page_table_cpu=table_cpu if packed else table_cpu[1:],
        )
        current_k = k[prefix:] if packed else k[-128:]
        current_v = v[prefix:] if packed else v[-128:]
        return backend.forward_extend(
            q if packed else q[-128:],
            current_k,
            current_v,
            layer,
            cache_locs if packed else cache_locs[-128:],
            pool,
            batch,
        )[-128:]

    alone = run(packed=False)
    with_neighbor = run(packed=True)
    torch.testing.assert_close(alone, with_neighbor, atol=0, rtol=0)

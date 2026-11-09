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

"""CPU numerical checks of the loaders used by dense Qwen3.5 TP."""

import pytest
import torch
import torch.nn.functional as F

from tokenspeed.runtime.layers.attention.configs.linear_attn import LinearAttnConfig
from tokenspeed.runtime.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.model_loader.weight_utils import (
    mamba_v2_sharded_weight_loader,
    sharded_weight_loader,
)


@pytest.mark.parametrize("widths", [(16, 16), (8, 8, 16, 16, 4, 4)])
def test_packed_projection_and_row_reduction(widths):
    generator = torch.Generator().manual_seed(43)
    x = torch.randn(5, 12, generator=generator)
    weights = [torch.randn(width, 12, generator=generator) for width in widths]
    shards = []
    for rank in range(2):
        layer = MergedColumnParallelLinear(
            12,
            list(widths),
            bias=False,
            params_dtype=torch.float32,
            tp_rank=rank,
            tp_size=2,
            tp_group=(0, 1),
        )
        for index, weight in enumerate(weights):
            layer.weight_loader(layer.weight, weight, index)
        local = F.linear(x, layer.weight).split([w // 2 for w in widths], dim=-1)
        shards.append(local)
    reconstructed = torch.cat(
        [
            torch.cat([shards[r][i] for r in range(2)], dim=-1)
            for i in range(len(widths))
        ],
        dim=-1,
    )
    torch.testing.assert_close(
        reconstructed, F.linear(x, torch.cat(weights)), atol=1e-6, rtol=1e-6
    )
    down = torch.randn(12, sum(widths), generator=generator)
    partials = []
    for rank in range(2):
        layer = RowParallelLinear(
            sum(widths),
            12,
            bias=False,
            reduce_results=False,
            params_dtype=torch.float32,
            tp_rank=rank,
            tp_size=2,
            tp_group=(0, 1),
        )
        layer.weight_loader(layer.weight, down)
        partials.append(F.linear(reconstructed.chunk(2, dim=-1)[rank], layer.weight))
    torch.testing.assert_close(
        sum(partials), F.linear(reconstructed, down), atol=2e-5, rtol=2e-5
    )


@pytest.mark.parametrize("kv_heads", [1, 2])
def test_gated_query_and_kv_head_partition(kv_heads):
    # Qwen packs Q and gate within each query head before the QKV loader.
    x = torch.arange(36, dtype=torch.float32).view(3, 12) / 36
    weights = [
        torch.arange(n * 12, dtype=torch.float32).view(n, 12) / 100
        for n in (8 * 2 * 4, kv_heads * 4, kv_heads * 4)
    ]
    for rank in range(2):
        layer = QKVParallelLinear(
            12,
            4,
            16,
            kv_heads,
            bias=False,
            params_dtype=torch.float32,
            tp_rank=rank,
            tp_size=2,
            tp_group=(0, 1),
        )
        for name, weight in zip(("q", "k", "v"), weights, strict=True):
            layer.weight_loader(layer.weight, weight, name)
        q, k, v = F.linear(x, layer.weight).split([32, 4, 4], dim=-1)
        expected_q = F.linear(x, weights[0]).view(3, 8, 2, 4)[
            :, rank * 4 : (rank + 1) * 4
        ]
        torch.testing.assert_close(q.view(3, 4, 2, 4), expected_q)
        for output, weight in zip((k, v), weights[1:], strict=True):
            expected = F.linear(x, weight)
            if kv_heads == 2:
                expected = expected.chunk(2, dim=-1)[rank]
            torch.testing.assert_close(output, expected)


def test_convolution_decay_and_state_head_alignment():
    widths = [8, 8, 16]
    weights = torch.arange(sum(widths) * 4, dtype=torch.float32).view(-1, 1, 4) / 100
    x = torch.arange(sum(widths) * 9, dtype=torch.float32).view(1, -1, 9) / 100
    reference = F.conv1d(x, weights, groups=sum(widths))
    decay = torch.arange(4, dtype=torch.float32)
    for rank in range(2):
        local_weight = torch.nn.Parameter(torch.empty(sum(widths) // 2, 1, 4))
        mamba_v2_sharded_weight_loader([(w, 0, False) for w in widths], 2, rank)(
            local_weight, weights
        )
        local_input = torch.cat(
            [part.chunk(2, dim=1)[rank] for part in x.split(widths, dim=1)], dim=1
        )
        expected = torch.cat(
            [part.chunk(2, dim=1)[rank] for part in reference.split(widths, dim=1)],
            dim=1,
        )
        torch.testing.assert_close(
            F.conv1d(local_input, local_weight, groups=16), expected
        )
        local_decay = torch.nn.Parameter(torch.empty(2))
        sharded_weight_loader(0, rank)(local_decay, decay)
        torch.testing.assert_close(local_decay, decay.chunk(2)[rank], rtol=0, atol=0)
    config = LinearAttnConfig(
        num_k_heads=2,
        num_v_heads=4,
        head_k_dim=4,
        head_v_dim=4,
        conv_kernel_size=4,
        layer_ids=(0,),
        tp_size=2,
    )
    assert config.conv_state_shape == (16, 3)
    assert config.temporal_state_shape == (2, 4, 4)


def test_vocabulary_shards_reconstruct_tied_embedding_and_logits():
    from tokenspeed.runtime.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    generator = torch.Generator().manual_seed(72)
    weight = torch.randn(128, 8, generator=generator)
    hidden = torch.randn(3, 8, generator=generator)
    tables = []
    logits = []
    for rank in range(2):
        embedding = VocabParallelEmbedding(
            128,
            8,
            params_dtype=torch.float32,
            tp_rank=rank,
            tp_size=2,
            tp_group=(0, 1),
            padding_size=64,
        )
        embedding.weight_loader(embedding.weight, weight)
        tables.append(embedding.weight)
        logits.append(F.linear(hidden, embedding.weight))
    reconstructed = torch.cat(tables)
    ids = torch.tensor([0, 63, 64, 127])
    torch.testing.assert_close(
        F.embedding(ids, reconstructed), F.embedding(ids, weight), rtol=0, atol=0
    )
    torch.testing.assert_close(torch.cat(logits, dim=-1), F.linear(hidden, weight))

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

"""Real two-rank HCCL checks of full-K weights and communication tails."""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytest.importorskip("torch_npu")

from tokenspeed_kernel.ops.gemm import mm

from tokenspeed.runtime.distributed.comm_backend import set_global_backend
from tokenspeed.runtime.distributed.comm_backend.hccl import HcclBackend
from tokenspeed.runtime.distributed.comm_manager import FullKCommManager
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.distributed.process_group_manager import process_group_manager
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
from tokenspeed.runtime.layers.linear import FullKParallelLinear

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available() or torch.npu.device_count() < 2,
    reason="Two visible Ascend chips required",
)


def _worker(rank, rendezvous):
    torch.npu.set_device(rank)
    dist.init_process_group(
        "hccl",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    group = (0, 1)
    process_group_manager.register_process_group("hccl", group, dist.group.WORLD)
    set_global_backend(HcclBackend())
    mapping = Mapping(rank=rank, world_size=2, attn_tp_size=2, dense_tp_size=2)
    manager = FullKCommManager(
        mapping, layer_id=0, input_layernorm=None, post_attn_layernorm=None
    )
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=1,
        num_extends=0,
        input_num_tokens=1,
        forward_mode=ForwardMode.DECODE,
        output_layout=ForwardOutputLayout(0, 0, 1, 1),
    )
    generator = torch.Generator().manual_seed(194)
    try:
        for dtype in (torch.bfloat16, torch.float16):
            for inner, width in ((1024, 1024), (6144, 5120), (17408, 5120)):
                weight = (
                    (torch.randn(width, inner, generator=generator) / inner**0.5)
                    .to(dtype)
                    .to(f"npu:{rank}")
                )
                with torch.device(f"npu:{rank}"):
                    layer = FullKParallelLinear(
                        inner,
                        width,
                        tp_rank=rank,
                        tp_size=2,
                        tp_group=group,
                        params_dtype=dtype,
                        prefix="projection",
                    )
                layer.weight_loader(layer.weight, weight)
                for rows in (0, 1, 17):
                    inputs = torch.randn(
                        rows, inner, generator=generator, dtype=dtype
                    ).to(f"npu:{rank}")
                    local = inputs.chunk(2, dim=-1)[rank].contiguous()
                    projected, _ = layer(local)
                    reference = mm(inputs, weight)
                    torch.testing.assert_close(
                        projected, reference.chunk(2, dim=-1)[rank], atol=0, rtol=0
                    )
                    if rows == 0:
                        continue
                    residual = torch.zeros_like(reference)
                    for tail in (manager.post_attn_comm, manager.post_dense_comm):
                        actual, returned_residual = tail(projected, residual, ctx)
                        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
                        torch.testing.assert_close(
                            returned_residual, residual, atol=0, rtol=0
                        )
        assert not manager.use_all_reduce_norm_fusion()
    finally:
        dist.destroy_process_group()


def test_full_k_projection_and_output_gathers(tmp_path):
    os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
    mp.spawn(_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True)

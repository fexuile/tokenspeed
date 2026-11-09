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

"""Exercise HCCL's tensor layouts using real CPU collectives without NPU mocks."""

from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tokenspeed.runtime.distributed.comm_backend.hccl import HcclBackend
from tokenspeed.runtime.distributed.process_group_manager import process_group_manager


def _worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    group = (0, 1)
    # The backend's layout logic is device-independent. Register the real CPU
    # transport under its group key; the separate NPU smoke test covers HCCL.
    process_group_manager.register_process_group("hccl", group, dist.group.WORLD)
    backend = HcclBackend()
    try:
        x = torch.full((2, 3), float(rank + 1))
        gathered = backend.all_gather(x, group, dim=1)
        torch.testing.assert_close(
            gathered, torch.tensor([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]]).expand(2, -1)
        )
        reduced = backend.reduce_scatter(x, group)
        torch.testing.assert_close(reduced, torch.full((1, 3), 3.0))
        keys = torch.tensor([-(1 << 63) + rank, (rank << 32) + 7, (1 << 63) - 1 - rank])
        backend.all_reduce(keys, group, op=dist.ReduceOp.MAX)
        torch.testing.assert_close(
            keys,
            torch.tensor([-(1 << 63) + 1, (1 << 32) + 7, (1 << 63) - 1]),
            rtol=0,
            atol=0,
        )
        counts = [1, 2]
        local = torch.full((counts[rank], 3), float(rank + 1))
        all_rows = backend.token_all_gather(local, group, counts)
        torch.testing.assert_close(
            all_rows, torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
        )
        own_rows = backend.token_reduce_scatter(all_rows, group, counts)
        torch.testing.assert_close(own_rows, local * 2)
    finally:
        dist.destroy_process_group()


def test_tensor_collectives_with_real_processes(tmp_path):
    mp.spawn(_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True)

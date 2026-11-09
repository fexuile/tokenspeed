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

"""Run with torchrun --standalone --nproc-per-node=2 on two visible NPUs."""

import os

import torch
import torch.distributed as dist
import torch_npu  # Registers the NPU device and HCCL backend.

from tokenspeed.runtime.distributed.comm_backend.hccl import HcclBackend
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.distributed.process_group_manager import process_group_manager


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 2 or torch.npu.device_count() != 2:
        raise ValueError("This smoke test requires exactly two visible chips and ranks")
    torch.npu.set_device(local_rank)
    mapping = Mapping(
        rank=rank,
        world_size=2,
        attn_tp_size=2,
        attn_dp_size=1,
        dense_tp_size=2,
        nprocs_per_node=2,
    )
    process_group_manager.init_distributed(mapping, backend="hccl", timeout=120)
    group = (0, 1)
    process_group_manager.init_process_group(group)
    backend = HcclBackend()
    try:
        values = torch.full(
            (17, 32), rank + 1, device=f"npu:{local_rank}", dtype=torch.bfloat16
        )
        backend.all_reduce(values, group)
        torch.testing.assert_close(
            values.cpu(), torch.full((17, 32), 3, dtype=torch.bfloat16), rtol=0, atol=0
        )
        keys = torch.tensor(
            [-(1 << 63), (rank << 32) + 7, (1 << 63) - 1 - rank],
            dtype=torch.int64,
            device=f"npu:{local_rank}",
        )
        backend.all_reduce(keys, group, op=dist.ReduceOp.MAX)
        torch.testing.assert_close(
            keys.cpu(),
            torch.tensor([-(1 << 63), (1 << 32) + 7, (1 << 63) - 1]),
            rtol=0,
            atol=0,
        )
        print(
            f"PASS rank={rank} pid={os.getpid()} device=npu:{local_rank} "
            f"visible={os.environ['ASCEND_RT_VISIBLE_DEVICES']} "
            f"allocated={torch.npu.memory_allocated()} reserved={torch.npu.memory_reserved()}",
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

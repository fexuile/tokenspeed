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

"""Profile each worker from its forward thread until orderly cache shutdown."""
import os
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from aics.hw2.communication import profile_workload as workload
import torch
import torch_npu
from tokenspeed.runtime.execution.model_executor import ModelExecutor

profiler = None
original_forward = ModelExecutor._forward_step

def forward(self, bs, ctx, sampling_info):
    global profiler
    if profiler is None:
        rank = torch.distributed.get_rank()
        directory = Path(os.environ['HW2_ANALYSIS_OUT']) / f'profile-rank{rank}'
        profiler = torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                str(directory), analyse_flag=False),
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level0))
        profiler.start()
        print(f'DIRECT_PROFILE_START rank={rank} pid={os.getpid()} device={torch.npu.current_device()}', flush=True)
    torch.npu.synchronize()
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    start_ns = time.time_ns()
    start_event.record()
    result = original_forward(self, bs, ctx, sampling_info)
    end_event.record()
    end_event.synchronize()
    row = dict(rank=torch.distributed.get_rank(), step=workload.step,
               tokens=ctx.input_num_tokens, bs=ctx.bs, extends=ctx.num_extends,
               start_epoch_ns=start_ns, end_epoch_ns=time.time_ns(),
               event_ms=start_event.elapsed_time(end_event),
               allocated_bytes=torch.npu.memory_allocated(),
               reserved_bytes=torch.npu.memory_reserved())
    directory = Path(os.environ['HW2_ANALYSIS_OUT'])
    with (directory / f"rank{row['rank']}-synchronized-forwards.jsonl").open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    return result

def stop():
    if profiler is not None:
        torch.npu.synchronize()
        profiler.stop()
        print(f'DIRECT_PROFILE_STOP rank={torch.distributed.get_rank()}', flush=True)

from tokenspeed.runtime.engine.event_loop import EventLoop
original_process_requests = EventLoop._process_new_requests
profile_stopped = False

def process_requests(self):
    global profile_stopped
    directory = Path(os.environ['HW2_ANALYSIS_OUT'])
    if not profile_stopped and (directory/'stop-profiles').exists():
        # The controller has paused scheduling; queue behind all prior work.
        self._device._thread.run(stop)
        profile_stopped = True
        rank = torch.distributed.get_rank()
        (directory/f'rank{rank}-profile-stopped').write_text('finalized\n')
    return original_process_requests(self)

ModelExecutor._forward_step = forward
EventLoop._process_new_requests = process_requests
original_run = workload.run
completed_cases = 0

async def run_and_finalize(engine, ids, batch, sampling):
    import asyncio
    global completed_cases
    result = await original_run(engine, ids, batch, sampling)
    completed_cases += 1
    if completed_cases == 8:
        manager = engine.tokenizer_manager
        assert await manager.pause_scheduler(mode='keep')
        assert await manager.is_scheduler_paused()
        directory = Path(os.environ['HW2_ANALYSIS_OUT'])
        (directory/'stop-profiles').write_text('workload complete; scheduler paused\n')
        async with asyncio.timeout(30):
            while not all((directory/f'rank{r}-profile-stopped').exists()
                          for r in range(int(os.environ['HW2_ANALYSIS_TP']))):
                await asyncio.sleep(0.01)
    return result

workload.run = run_and_finalize

if __name__ == '__main__':
    workload.main()

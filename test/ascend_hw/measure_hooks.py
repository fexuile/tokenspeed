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

"""Optional Event/HBM diagnostics printed by the scheduler, without files."""

import statistics
import time

_installed = False


def install(*, enabled):
    global _installed
    if not enabled or _installed:
        return
    from tokenspeed.runtime.execution.model_executor import ModelExecutor

    original = ModelExecutor._forward_step
    state = dict(input_tokens=0, decode_ms=[])

    def forward(self, bs, ctx, sampling_info):
        import torch

        if ctx.num_extends:
            state.update(input_tokens=ctx.input_num_tokens, decode_ms=[])
        if state["input_tokens"] == 0:
            return original(self, bs, ctx, sampling_info)
        torch.npu.synchronize()
        if ctx.num_extends:
            torch.npu.reset_peak_memory_stats()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        started = time.perf_counter()
        start.record()
        result = original(self, bs, ctx, sampling_info)
        end.record()
        end.synchronize()
        event_ms = start.elapsed_time(end)
        host_ms = (time.perf_counter() - started) * 1000
        if ctx.num_extends:
            phase = "prefill"
            count = 1
            mean = event_ms
        else:
            state["decode_ms"].append(event_ms)
            count = len(state["decode_ms"])
            if count not in (1, 8, 31, 32, 127, 128):
                return result
            phase = "decode"
            mean = statistics.mean(state["decode_ms"])
        print(
            f"NPU_DIAGNOSTIC input={state['input_tokens']} phase={phase} "
            f"steps={count} event_mean_ms={mean:.3f} last_sync_ms={host_ms:.3f} "
            f"peak_allocated_GiB={torch.npu.max_memory_allocated()/2**30:.4f} "
            f"peak_reserved_GiB={torch.npu.max_memory_reserved()/2**30:.4f}",
            flush=True,
        )
        return result

    ModelExecutor._forward_step = forward
    _installed = True

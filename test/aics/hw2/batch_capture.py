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

"""Record worker batch sizes for TP correctness runs without device synchronization."""

import json
from pathlib import Path

_installed = False


def install(output_dir):
    global _installed
    if _installed:
        return
    import torch

    from tokenspeed.runtime.execution.model_executor import ModelExecutor

    original = ModelExecutor._forward_step

    def forward(self, bs, ctx, sampling_info):
        result = original(self, bs, ctx, sampling_info)
        rank = torch.distributed.get_rank()
        event = dict(
            bs=ctx.bs, num_extends=ctx.num_extends, tokens=ctx.input_num_tokens
        )
        with (Path(output_dir) / f"rank{rank}.jsonl").open("a") as stream:
            stream.write(f"{json.dumps(event)}\n")
        return result

    ModelExecutor._forward_step = forward
    _installed = True

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
from transformers import PretrainedConfig

pytest.importorskip("torch_npu")

from tokenspeed.runtime.layers.logits_processor import LogitsProcessor
from tokenspeed.runtime.layers.vocab_parallel_embedding import ParallelLMHead

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend logits tests require an NPU"
)


def test_lm_head_logits_do_not_depend_on_other_rows():
    torch.manual_seed(83)
    hidden = torch.randn(2176, 2560, device="npu", dtype=torch.bfloat16)
    head = ParallelLMHead(
        12352,
        2560,
        params_dtype=torch.bfloat16,
        padding_size=1,
        tp_rank=0,
        tp_size=1,
        tp_group=(0,),
    ).to("npu")
    with torch.no_grad():
        head.weight.copy_(torch.randn_like(head.weight) / 2560**0.5)
    processor = LogitsProcessor(
        PretrainedConfig(vocab_size=12352),
        dp_lm_head_tp=False,
    )
    packed = processor._get_logits(hidden, head, None, require_full_vocab=True)
    for start in (0, 7, 2048):
        single = processor._get_logits(
            hidden[start : start + 128], head, None, require_full_vocab=True
        )
        torch.testing.assert_close(single, packed[start : start + 128], atol=0, rtol=0)

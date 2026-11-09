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

"""Cache-off requests retain the history their own forwards computed."""

import pytest
import tokenspeed_scheduler as ts
from conftest import _advance, _find_forward_op, _finish, _spec


@pytest.mark.parametrize("chunk_tokens", [2, 16])
@pytest.mark.parametrize("prompt_tokens", [4, 5, 8])
@pytest.mark.parametrize("with_state", [False, True])
@pytest.mark.parametrize("host_cache", [False, True])
def test_disabled_prefix_cache_keeps_computed_history(
    chunk_tokens: int, prompt_tokens: int, with_state: bool, host_cache: bool
) -> None:
    cfg = ts.SchedulerConfig()
    cfg.prefix_granularity = 4
    cfg.num_device_pages = 65
    cfg.num_snapshot_pages = 1
    cfg.num_host_pages = 65 if host_cache else 0
    cfg.max_scheduled_tokens = chunk_tokens
    cfg.max_batch_size = 1
    cfg.disable_l2_cache = not host_cache
    cfg.disable_prefix_cache = True
    cfg.enable_kv_cache_events = True
    groups = [
        ts.CacheGroupConfig(
            group_id="full",
            block_granularity=4,
            total_pages=65,
            attn_kind=ts.AttnKind.Full,
        )
    ]
    if with_state:
        groups.append(
            ts.CacheGroupConfig(
                group_id="state",
                block_granularity=0,
                total_pages=65,
                attn_kind=ts.AttnKind.State,
            )
        )
    cfg.cache_groups = groups
    scheduler = ts.Scheduler(cfg)

    def submit_and_prefill(request_id: str, count: int) -> list[int]:
        spec = _spec(request_id, list(range(count)))
        spec.max_new_tokens = 4
        scheduler.submit_requests([spec])
        computed = 0
        history = []
        while computed < count:
            plan = scheduler.next_execution_plan()
            assert not list(plan.cache)
            batch = _find_forward_op(plan)
            assert batch is not None
            assert list(batch.request_ids) == [request_id]
            assert list(batch.extend_prefix_lens) == [computed]
            row = list(batch.block_tables["full"][0])
            # A completed chunk must not switch to another request's cache.
            assert row[: len(history)] == history
            computed += batch.input_lengths[0]
            history = row[: (computed + 3) // 4]
            _advance(scheduler, request_id, [101] if computed == count else [])
        return history

    submit_and_prefill("seed", 12)
    _finish(scheduler, "seed")
    scheduler.next_execution_plan()

    history = submit_and_prefill("repeat", prompt_tokens)
    for token in range(3):
        batch = _find_forward_op(scheduler.next_execution_plan())
        assert batch is not None
        # Cache publication/deduplication must not replace computed prompt KV,
        # including at the first decode or a later page boundary.
        assert list(batch.block_tables["full"][0])[: len(history)] == history
        _advance(scheduler, "repeat", [102 + token])
    _finish(scheduler, "repeat")
    assert not list(scheduler.next_execution_plan().cache)
    assert scheduler.active_lcm_blocks() == 0
    assert not scheduler.drain_kv_events()

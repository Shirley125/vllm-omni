# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.v1.core.sched.scheduler import Scheduler as VLLMScheduler

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_ar_scheduler_restores_queues_when_super_schedule_raises(monkeypatch):
    scheduler = object.__new__(OmniARScheduler)
    scheduler.chunk_transfer_adapter = MagicMock()
    scheduler.waiting = object()
    scheduler.running = object()

    def _raise_schedule(_self):
        raise RuntimeError("schedule failed")

    monkeypatch.setattr(VLLMScheduler, "schedule", _raise_schedule)

    with pytest.raises(RuntimeError, match="schedule failed"):
        OmniARScheduler.schedule(scheduler)

    adapter = scheduler.chunk_transfer_adapter
    adapter.process_pending_chunks.assert_called_once_with(scheduler.waiting, scheduler.running)
    adapter.restore_queues.assert_called_once_with(scheduler.waiting, scheduler.running)


def test_ar_scheduler_calls_postprocess_on_success(monkeypatch):
    scheduler = object.__new__(OmniARScheduler)
    scheduler.chunk_transfer_adapter = MagicMock()
    scheduler.waiting = []
    scheduler.running = []
    scheduler.requests = {}
    scheduler.get_finished_requests_needing_kv_transfer = lambda: {"req-1": {"seq_len": 1, "block_ids": []}}

    super_output = SimpleNamespace(scheduled_new_reqs=[], base_field="ok")
    monkeypatch.setattr(VLLMScheduler, "schedule", lambda _self: super_output)

    import vllm_omni.core.sched.omni_ar_scheduler as ar_mod

    class _FakeSchedulerOutput:
        __dataclass_fields__ = {"base_field": object()}

    monkeypatch.setattr(ar_mod, "SchedulerOutput", _FakeSchedulerOutput)
    monkeypatch.setattr(ar_mod, "OmniSchedulerOutput", lambda **kwargs: kwargs)

    out = OmniARScheduler.schedule(scheduler)

    adapter = scheduler.chunk_transfer_adapter
    adapter.process_pending_chunks.assert_called_once_with(scheduler.waiting, scheduler.running)
    adapter.restore_queues.assert_called_once_with(scheduler.waiting, scheduler.running)
    adapter.postprocess_scheduler_output.assert_called_once_with(super_output, scheduler.requests)
    assert out["finished_requests_needing_kv_transfer"] == {"req-1": {"seq_len": 1, "block_ids": []}}


def test_generation_scheduler_fallback_restores_and_postprocesses(monkeypatch):
    scheduler = object.__new__(OmniGenerationScheduler)
    scheduler.chunk_transfer_adapter = MagicMock()
    scheduler.max_num_scheduled_tokens = 0
    scheduler.policy = "fcfs"
    scheduler.waiting = []
    scheduler.running = []
    scheduler.max_num_running_reqs = 4

    import vllm_omni.core.sched.omni_generation_scheduler as gen_mod

    monkeypatch.setattr(gen_mod, "create_request_queue", lambda _policy: [])
    sentinel_output = object()
    monkeypatch.setattr(VLLMScheduler, "schedule", lambda _self: sentinel_output)

    out = OmniGenerationScheduler.schedule(scheduler)

    adapter = scheduler.chunk_transfer_adapter
    adapter.process_pending_chunks.assert_called_once_with(scheduler.waiting, scheduler.running)
    adapter.restore_queues.assert_called_once_with(scheduler.waiting, scheduler.running)
    adapter.postprocess_scheduler_output.assert_called_once_with(sentinel_output)
    assert out is sentinel_output

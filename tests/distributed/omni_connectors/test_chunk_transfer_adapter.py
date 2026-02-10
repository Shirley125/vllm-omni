# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.v1.request import RequestStatus

from vllm_omni.distributed.omni_connectors.transfer_adapter.base import OmniTransferAdapterBase
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import (
    OmniChunkTransferAdapter,
)
from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec


class DummyWaitingQueue(list):
    def prepend_requests(self, requests):
        self[:0] = list(requests)

    def add_request(self, request):
        self.append(request)


def make_request(
    request_id: str,
    status: RequestStatus,
    *,
    external_req_id: str | None = None,
    prompt_token_ids=None,
    num_computed_tokens: int = 0,
):
    if prompt_token_ids is None:
        prompt_token_ids = []
    return SimpleNamespace(
        request_id=request_id,
        external_req_id=external_req_id or request_id,
        status=status,
        prompt_token_ids=prompt_token_ids,
        num_computed_tokens=num_computed_tokens,
        additional_information=None,
    )


@pytest.fixture
def adapter_builder(monkeypatch):
    def _build(*, stage_id: int = 1, model_mode: str = "ar", max_num_seqs: int = 2):
        connector = MagicMock()
        connector.stage_id = stage_id
        connector.get.return_value = None
        connector.put.return_value = (True, 1, {})

        def _fake_base_init(self, config):
            self.config = config
            if not hasattr(self, "connector"):
                self.connector = None
            self._pending_load_reqs = {}
            self._finished_load_reqs = set()
            self._pending_save_reqs = {}
            self._finished_save_reqs = set()
            self.stop_event = threading.Event()
            self.lock = threading.Lock()

        monkeypatch.setattr(OmniTransferAdapterBase, "__init__", _fake_base_init)
        monkeypatch.setattr(
            OmniChunkTransferAdapter,
            "create_connector",
            classmethod(lambda cls, _model_config: connector),
        )

        model_config = SimpleNamespace(worker_type=model_mode)
        scheduler_config = SimpleNamespace(max_num_seqs=max_num_seqs)
        vllm_config = SimpleNamespace(model_config=model_config, scheduler_config=scheduler_config)
        adapter = OmniChunkTransferAdapter(vllm_config)
        return adapter, connector

    return _build


def test_create_connector_with_default_config(monkeypatch):
    captured = {}

    def _fake_create(spec):
        captured["spec"] = spec
        return "connector_obj"

    monkeypatch.setattr(
        "vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter"
        ".OmniConnectorFactory.create_connector",
        _fake_create,
    )

    model_config = SimpleNamespace()
    connector = OmniChunkTransferAdapter.create_connector(model_config)

    assert connector == "connector_obj"
    assert isinstance(captured["spec"], ConnectorSpec)
    assert captured["spec"].name == "SharedMemoryConnector"
    assert captured["spec"].extra == {}


def test_create_connector_with_object_config(monkeypatch):
    captured = {}

    def _fake_create(spec):
        captured["spec"] = spec
        return "connector_obj"

    monkeypatch.setattr(
        "vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter"
        ".OmniConnectorFactory.create_connector",
        _fake_create,
    )

    cfg_obj = SimpleNamespace(name="YuanrongConnector", extra={"key": "value"})
    model_config = SimpleNamespace(stage_connector_config=cfg_obj)

    connector = OmniChunkTransferAdapter.create_connector(model_config)

    assert connector == "connector_obj"
    assert isinstance(captured["spec"], ConnectorSpec)
    assert captured["spec"].name == "YuanrongConnector"
    assert captured["spec"].extra == {"key": "value"}


def test_load_async_stage_zero_is_noop(adapter_builder):
    adapter, _ = adapter_builder(stage_id=0)
    request = SimpleNamespace(request_id="req-0", external_req_id="external-0")

    adapter.load_async(request)

    assert adapter.request_ids_mapping["req-0"] == "external-0"
    assert adapter._pending_load_reqs == {}
    assert not hasattr(request, "additional_information")


def test_load_async_stage_nonzero_registers_request(adapter_builder):
    adapter, _ = adapter_builder(stage_id=1)
    request = SimpleNamespace(request_id="req-1", external_req_id="external-1")

    adapter.load_async(request)

    assert request.additional_information is None
    assert adapter._pending_load_reqs["req-1"] is request
    assert adapter.request_ids_mapping["req-1"] == "external-1"


def test_save_async_enqueues_task_and_increments_chunk_id(adapter_builder):
    adapter, _ = adapter_builder(stage_id=1)
    request = SimpleNamespace(external_req_id="external-1")
    payload = {"hidden_states": torch.tensor([[1.0]]), "finished": False}
    adapter.custom_process_next_stage_input_func = lambda **kwargs: payload

    adapter.save_async(pooling_output=torch.tensor([1.0]), request=request)
    adapter.save_async(pooling_output=torch.tensor([2.0]), request=request)

    assert adapter.put_req_chunk["external-1"] == 2
    tasks = adapter._pending_save_reqs["external-1"]
    assert len(tasks) == 2
    assert tasks[0]["put_key"] == "external-1_1_0"
    assert tasks[1]["put_key"] == "external-1_1_1"


def test_save_async_ignores_empty_payload(adapter_builder):
    adapter, _ = adapter_builder(stage_id=1)
    request = SimpleNamespace(external_req_id="external-empty")
    adapter.custom_process_next_stage_input_func = lambda **kwargs: {}

    adapter.save_async(pooling_output=None, request=request)

    assert adapter.put_req_chunk["external-empty"] == 0
    assert "external-empty" not in adapter._pending_save_reqs


def test_poll_single_request_ar_updates_payload(adapter_builder):
    adapter, connector = adapter_builder(stage_id=2, model_mode="ar")
    request = make_request("req-ar", RequestStatus.WAITING, external_req_id="external-ar")
    adapter._pending_load_reqs["req-ar"] = request
    adapter.request_ids_mapping["req-ar"] = "external-ar"

    payload = {
        "code_predictor_codes": [[1, 2]],
        "hidden_states": torch.tensor([[3.0]]),
        "finished": True,
    }
    connector.get.return_value = (payload, 128)

    adapter._poll_single_request("req-ar")

    connector.get.assert_called_once_with("1", "2", "external-ar_1_0")
    assert request.additional_information == payload
    assert adapter.get_req_chunk["req-ar"] == 1
    assert "req-ar" in adapter._finished_load_reqs
    assert "req-ar" in adapter.finished_requests
    assert "req-ar" not in adapter._pending_load_reqs


def test_poll_single_request_non_ar_marks_request_finished(adapter_builder):
    adapter, connector = adapter_builder(stage_id=2, model_mode="diffusion")
    request = make_request("req-diff", RequestStatus.WAITING, external_req_id="external-diff", num_computed_tokens=9)
    adapter._pending_load_reqs["req-diff"] = request
    adapter.request_ids_mapping["req-diff"] = "external-diff"

    payload = {"code_predictor_codes": [[7, 8, 9]], "finished": True}
    connector.get.return_value = (payload, 64)

    adapter._poll_single_request("req-diff")

    assert request.prompt_token_ids == [[7, 8, 9]]
    assert request.num_computed_tokens == 0
    assert request.status == RequestStatus.FINISHED_STOPPED
    assert "req-diff" in adapter._finished_load_reqs
    assert "req-diff" in adapter.finished_requests


def test_update_request_payload_concatenates_tensor_and_list(adapter_builder):
    adapter, _ = adapter_builder()
    first_payload = {"hidden_states": torch.tensor([[1.0]]), "codes": [1], "finished": False}
    second_payload = {"hidden_states": torch.tensor([[2.0]]), "codes": [2], "finished": True}

    adapter._update_request_payload("external-1", first_payload)
    updated = adapter._update_request_payload("external-1", second_payload)

    assert torch.equal(updated["hidden_states"], torch.tensor([[1.0], [2.0]]))
    assert updated["codes"] == [1, 2]
    assert updated["finished"] is True


def test_process_pending_chunks_moves_pending_requests(adapter_builder):
    adapter, _ = adapter_builder(stage_id=1, max_num_seqs=8)
    waiting_request = make_request("wait-1", RequestStatus.WAITING)
    running_request = make_request("run-1", RequestStatus.RUNNING)
    waiting_queue = DummyWaitingQueue([waiting_request])
    running_queue = [running_request]

    adapter.process_pending_chunks(waiting_queue, running_queue)

    assert waiting_request.status == RequestStatus.WAITING_FOR_CHUNK
    assert running_request.status == RequestStatus.WAITING_FOR_CHUNK
    assert list(adapter.waiting_for_chunk_waiting_requests) == [waiting_request]
    assert list(adapter.waiting_for_chunk_running_requests) == [running_request]
    assert waiting_queue == []
    assert running_queue == []
    assert "wait-1" in adapter._pending_load_reqs
    assert "run-1" in adapter._pending_load_reqs


def test_process_pending_chunks_limits_running_queue(adapter_builder):
    adapter, _ = adapter_builder(stage_id=1, max_num_seqs=2)
    req_1 = make_request("run-ready-1", RequestStatus.WAITING_FOR_CHUNK)
    req_2 = make_request("run-ready-2", RequestStatus.WAITING_FOR_CHUNK)
    req_3 = make_request("run-keep", RequestStatus.RUNNING)
    adapter._finished_load_reqs = {"run-ready-1", "run-ready-2"}
    adapter.requests_with_ready_chunks = {"run-keep"}

    waiting_queue = DummyWaitingQueue()
    running_queue = [req_1, req_2, req_3]

    adapter.process_pending_chunks(waiting_queue, running_queue)

    assert req_1.status == RequestStatus.RUNNING
    assert req_2.status == RequestStatus.RUNNING
    assert running_queue == [req_1, req_2]
    assert waiting_queue == [req_3]


def test_restore_queues_puts_requests_back(adapter_builder):
    adapter, _ = adapter_builder()
    wait_1 = make_request("wait-1", RequestStatus.WAITING_FOR_CHUNK)
    wait_2 = make_request("wait-2", RequestStatus.WAITING_FOR_CHUNK)
    run_1 = make_request("run-1", RequestStatus.WAITING_FOR_CHUNK)
    waiting_queue = DummyWaitingQueue()
    running_queue = []

    adapter.waiting_for_chunk_waiting_requests = deque([wait_1, wait_2])
    adapter.waiting_for_chunk_running_requests = deque([run_1])

    adapter.restore_queues(waiting_queue, running_queue)

    assert waiting_queue == [wait_1, wait_2]
    assert running_queue == [run_1]
    assert adapter.waiting_for_chunk_waiting_requests == deque()
    assert adapter.waiting_for_chunk_running_requests == deque()


def test_postprocess_scheduler_output_attaches_cache_and_clears_ready(adapter_builder):
    adapter, _ = adapter_builder()
    adapter.requests_with_ready_chunks = {"new-ready", "cached-ready", "leftover"}
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="new-ready"), SimpleNamespace(req_id="new-other")],
        scheduled_cached_reqs=SimpleNamespace(req_ids=["cached-ready", "cached-missing"]),
    )
    requests = {
        "cached-ready": SimpleNamespace(additional_information={"k": "v"}),
    }

    adapter.postprocess_scheduler_output(scheduler_output, requests)

    cached_info = scheduler_output.scheduled_cached_reqs.additional_information
    assert cached_info["cached-ready"] == {"k": "v"}
    assert cached_info["cached-missing"] is None
    assert adapter.requests_with_ready_chunks == {"leftover"}


def test_get_finished_requests_clears_internal_state(adapter_builder):
    adapter, _ = adapter_builder()
    adapter._finished_load_reqs = {"r1", "r2"}

    finished = adapter.get_finished_requests()

    assert finished == {"r1", "r2"}
    assert adapter._finished_load_reqs == set()

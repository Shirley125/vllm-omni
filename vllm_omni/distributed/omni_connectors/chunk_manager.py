# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import torch
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request, RequestStatus

from .utils.logging import get_connector_logger

logger = get_connector_logger(__name__)


class BasicOmniTransferManager:
    """Common lifecycle helpers for Omni transfer managers."""

    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

    def close(self, timeout: float | None = None) -> None:
        """Signal background loops to stop and join threads."""
        self.stop_event.set()
        for thread in self._get_threads():
            if thread and thread.is_alive():
                thread.join(timeout=timeout)

    def _get_threads(self) -> list[threading.Thread]:
        return []


class OmniChunkTransferManager(BasicOmniTransferManager):
    """Manages asynchronous retrieval of chunks via OmniConnector."""

    def __init__(self, connector: Any):
        super().__init__()
        self.connector = connector
        # Requests that are waiting to be polled
        self.pending_load_reqs: dict[str, Request] = {}
        # Requests that have successfully retrieved a chunk
        self.finished_load_reqs: set[str] = set()

        # Requests that are waiting to be saved
        self.pending_save_reqs: dict[str, deque[dict[str, Any]]] = {}
        # Requests that have successfully saved a chunk
        self.finished_save_reqs: set[str] = set()

        self._waiting_for_chunk_waiting_requests: deque[Request] = deque()
        self._waiting_for_chunk_running_requests: deque[Request] = deque()
        self._finished_load_chunk_reqs: set[str] = set()
        self._requests_with_ready_chunks: set[str] = set()

        self.recv_thread = threading.Thread(target=self.recv_loop, daemon=True)
        self.recv_thread.start()

        self.save_thread = threading.Thread(target=self.save_loop, daemon=True)
        self.save_thread.start()

    def _get_threads(self) -> list[threading.Thread]:
        return [self.recv_thread, self.save_thread]

    # ------------------------------------------------------------------
    # Scheduler helper methods
    # ------------------------------------------------------------------
    def process_pending_chunks(self, waiting: Any, running: Any) -> int:
        stage_id = getattr(self.connector, "stage_id", None)
        if self.connector is None or stage_id in (None, 0):
            return 0

        self._finished_load_chunk_reqs = self.get_finished_load_requests()
        self._process_chunk_queue(waiting, self._waiting_for_chunk_waiting_requests, RequestStatus.WAITING)
        self._process_chunk_queue(running, self._waiting_for_chunk_running_requests, RequestStatus.RUNNING)
        return len(self._waiting_for_chunk_running_requests)

    def restore_queues(self, waiting: Any, running: Any) -> None:
        for request in self._waiting_for_chunk_waiting_requests:
            waiting.add_request(request)
        self._waiting_for_chunk_waiting_requests = deque()

        if self._waiting_for_chunk_running_requests:
            running.extend(self._waiting_for_chunk_running_requests)
        self._waiting_for_chunk_running_requests = deque()

        self._finished_load_chunk_reqs = set()

    def filter_scheduler_output(self, scheduler_output: SchedulerOutput) -> None:
        if scheduler_output.scheduled_new_reqs:
            for req_data in scheduler_output.scheduled_new_reqs:
                if req_data.req_id in self._requests_with_ready_chunks:
                    self._requests_with_ready_chunks.remove(req_data.req_id)

        if scheduler_output.scheduled_cached_reqs:
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                if req_id in self._requests_with_ready_chunks:
                    self._requests_with_ready_chunks.remove(req_id)

    # ------------------------------------------------------------------
    # Async IO methods
    # ------------------------------------------------------------------
    def request_chunk(self, request: Request) -> None:
        stage_id = self.connector.stage_id
        request_id = request.request_id
        self.connector.request_ids_mapping[request_id] = request.external_req_id

        if stage_id == 0:
            return
        if not hasattr(request, "additional_information"):
            request.additional_information = None
        with self.lock:
            self.pending_load_reqs[request_id] = request

    def submit_chunk(
        self,
        pooling_output: Any,
        request: Request,
        custom_process_input_func: Callable[[dict[str, Any], Request], dict[str, Any]] | None = None,
    ) -> None:
        stage_id = self.connector.stage_id
        next_stage_id = stage_id + 1
        request_id = request.external_req_id

        # Snapshot prompt_token_ids
        prompt_token_ids = list(request.prompt_token_ids)
        self.connector.request_prompt_token_ids[request_id] = prompt_token_ids
        chunk_id = self.connector.put_requests[request_id]

        # Process payload in main thread to avoid race conditions on request state
        payload_data = None
        if custom_process_input_func:
            try:
                payload_data = custom_process_input_func(
                    pooling_output=pooling_output,
                    request=request,
                )

            except Exception as e:
                logger.error(f"Failed to use custom_process_input_func for payload extraction: {e}")

            if not payload_data:
                logger.warning(f"[Stage-{stage_id}] No payload data to send for request {request_id}")
                return
            if stage_id == 0 and chunk_id == 0:
                if self.connector.request_payload.get(request_id) is None:
                    if not payload_data.get("finished"):
                        self.connector.request_payload[request_id] = payload_data
                        return
                else:
                    save_payload = self.connector.request_payload.pop(request_id)
                    payload_data["thinker_embeddings"] = torch.cat(
                        (save_payload.get("thinker_embeddings"), payload_data.get("thinker_embeddings")), dim=0
                    )
                    payload_data["thinker_hidden_states"] = torch.cat(
                        (save_payload.get("thinker_hidden_states"), payload_data.get("thinker_hidden_states")), dim=0
                    )
                    logger.info(f"[Stage-{stage_id}] Merged embeddings and hidden states for request {request_id}")

            if stage_id == 1:
                # TODO: Make parameters configurable and optimize algorithms
                chunk_size = left_context_size = 25
                self.connector.code_prompt_token_ids[request_id].append(payload_data.get("code_predictor_codes", []))
                length = len(self.connector.code_prompt_token_ids[request_id])
                chunk_length = length % chunk_size
                if chunk_length != 0 and not payload_data.get("finished"):
                    return

                context_length = chunk_length if chunk_length != 0 else chunk_size
                end_index = min(length, left_context_size + context_length)
                payload_data["code_predictor_codes"] = (
                    torch.tensor(self.connector.code_prompt_token_ids[request_id][-end_index:])
                    .transpose(0, 1)
                    .reshape(-1)
                    .tolist()
                )

        # Increment chunk_id here since we are committing to send
        self.connector.put_requests[request_id] += 1
        connector_put_key = f"{request.external_req_id}_{stage_id}_{chunk_id}"

        task = {
            "stage_id": stage_id,
            "next_stage_id": next_stage_id,
            "put_key": connector_put_key,
            "data": payload_data,
            "request_id": request_id,
        }

        with self.lock:
            if request_id not in self.pending_save_reqs:
                self.pending_save_reqs[request_id] = deque()
            self.pending_save_reqs[request_id].append(task)

    def get_finished_load_requests(self) -> set[str]:
        with self.lock:
            finished_load = set(self.finished_load_reqs)
            self.finished_load_reqs = set()
        return finished_load

    # ------------------------------------------------------------------
    # Internal loop methods
    # ------------------------------------------------------------------
    def recv_loop(self) -> None:
        while not self.stop_event.is_set():
            # Iterate over a snapshot of pending requests
            with self.lock:
                pending_reqs_ids = list(self.pending_load_reqs.keys())

            for req_id in pending_reqs_ids:
                stage_id = self.connector.stage_id
                target_stage_id = stage_id - 1
                chunk_id = self.connector.get_requests[req_id]
                external_req_id = self.connector.request_ids_mapping.get(req_id, req_id)
                connector_get_key = f"{external_req_id}_{target_stage_id}_{chunk_id}"

                try:
                    # Use timeout=0 for non-blocking poll
                    payload_data, size = self.connector.get(
                        str(target_stage_id),
                        str(stage_id),
                        connector_get_key,
                    )

                    if payload_data:
                        logger.debug(f"[Stage-{stage_id}] Received payload {payload_data}")

                        self.connector.request_prompt_token_ids[req_id] = payload_data.get("thinker_input_ids", [])
                        # Update connector state
                        self.connector.get_requests[req_id] += 1
                        req = self.pending_load_reqs[req_id]
                        # todo: just for qwen3_omni?
                        if stage_id != 2:
                            req.additional_information = payload_data
                            if payload_data.get("finished"):
                                self.connector.finished_requests.add(req_id)
                        else:
                            if payload_data.get("finished"):
                                self.connector.finished_requests.add(req_id)
                                req.status = RequestStatus.FINISHED_STOPPED

                            # TODO: remove special handling for prompt token ids ?
                            if chunk_id == 0:
                                req.prompt_token_ids = payload_data.get("code_predictor_codes", [])
                            else:
                                req.prompt_token_ids += payload_data.get("code_predictor_codes", [])

                        # Mark as finished for consumption
                        with self.lock:
                            self.finished_load_reqs.add(req_id)
                            if req_id in self.pending_load_reqs:
                                del self.pending_load_reqs[req_id]
                        logger.info(f"[Stage-{stage_id}] Received one chunk for key {connector_get_key}")
                except Exception as e:
                    logger.warning(f"[Stage-{stage_id}] Receiving chunk with error {e}")
                    pass

            time.sleep(0.001)

    def save_loop(self) -> None:
        while not self.stop_event.is_set():
            task = None
            with self.lock:
                pending_save_reqs_ids = list(self.pending_save_reqs.keys())
                for req_id in pending_save_reqs_ids:
                    if self.pending_save_reqs[req_id]:
                        task = self.pending_save_reqs[req_id].popleft()
                        if not self.pending_save_reqs[req_id]:
                            del self.pending_save_reqs[req_id]
                        break

            if task:
                connector_put_key = task["put_key"]
                stage_id = task["stage_id"]
                next_stage_id = task["next_stage_id"]
                payload_data = task["data"]
                request_id = task["request_id"]

                try:
                    success, size, metadata = self.connector.put(
                        from_stage=str(stage_id),
                        to_stage=str(next_stage_id),
                        put_key=connector_put_key,
                        data=payload_data,
                    )

                    if success:
                        logger.info(f"[Stage-{stage_id}] Sent {connector_put_key}")
                        with self.lock:
                            self.finished_save_reqs.add(request_id)

                except Exception as e:
                    logger.error(f"[Stage-{stage_id}] Error in save_loop for key {connector_put_key}: {e}")
            else:
                time.sleep(0.001)

    # ------------------------------------------------------------------
    # Backward-compatible aliases
    # ------------------------------------------------------------------
    def get_finished(self) -> set[str]:
        return self.get_finished_load_requests()

    def get_chunk(self, request: Request) -> None:
        self.request_chunk(request)

    def put_chunk(
        self,
        pooling_output: Any,
        request: Request,
        custom_process_input_func: Callable[[dict[str, Any], Request], dict[str, Any]] | None = None,
    ) -> None:
        self.submit_chunk(pooling_output, request, custom_process_input_func)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _process_chunk_queue(
        self,
        queue: Any,
        waiting_for_chunk_list: deque[Request],
        target_status: RequestStatus,
    ) -> None:
        queue_snapshot = list(queue)
        for request in queue_snapshot:
            if request.status != RequestStatus.WAITING_FOR_CHUNK:
                if request.request_id in self._requests_with_ready_chunks:
                    continue
                if request.request_id in self.connector.finished_requests:
                    request.additional_information = None
                    continue
                self.request_chunk(request)
                request.status = RequestStatus.WAITING_FOR_CHUNK
            else:
                if request.request_id in self._finished_load_chunk_reqs:
                    request.status = target_status
                    self._requests_with_ready_chunks.add(request.request_id)
                    continue
            queue.remove(request)
            waiting_for_chunk_list.append(request)

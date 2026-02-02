# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time
from collections import defaultdict, deque
from typing import Any

import torch
from vllm.v1.request import RequestStatus

from .utils.logging import get_connector_logger

logger = get_connector_logger(__name__)


class BasicOmniTransferManager:
    """Base class for managing asynchronous data transfer via OmniConnector.
    
    This class handles the core loop logic and connector interactions, but
    leaves the specific data processing (chunks, KV cache, etc.) to subclasses.
    """
    
    def __init__(self, connector):
        self.connector = connector
        # Requests that are waiting to be polled
        self._pending_load_reqs = {}
        # Requests that have successfully retrieved data
        self._finished_load_reqs = set()

        # Requests that are waiting to be saved
        self._pending_save_reqs = {}
        # Requests that have successfully saved data
        self._finished_save_reqs = set()

        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        
        self.recv_thread = threading.Thread(target=self.recv_loop, daemon=True)
        self.recv_thread.start()

        self.save_thread = threading.Thread(target=self.save_loop, daemon=True)
        self.save_thread.start()

    def recv_loop(self):
        """Loop to poll for incoming data."""
        while not self.stop_event.is_set():
            # Iterate over a snapshot of pending requests
            with self.lock:
                pending_reqs_ids = list(self._pending_load_reqs.keys())

            for req_id in pending_reqs_ids:
                try:
                    self._process_single_recv(req_id)
                except Exception as e:
                    logger.warning(f"Error receiving data for {req_id}: {e}")
                    pass

            time.sleep(0.001)

    def save_loop(self):
        """Loop to send outgoing data."""
        while not self.stop_event.is_set():
            task = None
            with self.lock:
                pending_save_reqs_ids = list(self._pending_save_reqs.keys())
                for req_id in pending_save_reqs_ids:
                    if self._pending_save_reqs[req_id]:
                        task = self._pending_save_reqs[req_id].popleft()
                        if not self._pending_save_reqs[req_id]:
                            del self._pending_save_reqs[req_id]
                        break

            if task:
                try:
                    self._process_single_save(task)
                except Exception as e:
                    logger.error(f"Error saving data for {task.get('request_id')}: {e}")
            else:
                time.sleep(0.001)

    def _process_single_recv(self, req_id: str):
        """Process a single receive attempt. To be implemented by subclasses if needed,
        or use a generic implementation."""
        raise NotImplementedError

    def _process_single_save(self, task: dict):
        """Process a single save attempt. To be implemented by subclasses if needed,
        or use a generic implementation."""
        raise NotImplementedError

    def get_finished_load_requests(self):
        with self.lock:
            finished_load = set(self._finished_load_reqs)
            self._finished_load_reqs = set()
        return finished_load


class OmniChunkManager(BasicOmniTransferManager):
    """Manages asynchronous retrieval and storage of data chunks via OmniConnector."""

    def __init__(self, connector):
        super().__init__(connector)
        
        # State specific to Chunk management
        self.put_requests: dict[str, int] = defaultdict(int)
        self.get_requests: dict[str, int] = defaultdict(int)
        self.finished_requests: set[str] = set()
        self.request_payload = {}
        self.request_prompt_token_ids: dict[str, list[int]] = defaultdict(list)
        self.code_prompt_token_ids: dict[str, list[list[int]]] = defaultdict(list)

        self.waiting_for_chunk_waiting_requests: deque[Any] = deque()
        self.waiting_for_chunk_running_requests: deque[Any] = deque()
        self.requests_with_ready_chunks = set()

    def _process_single_recv(self, req_id: str):
        stage_id = self.connector.stage_id
        target_stage_id = stage_id - 1
        chunk_id = self.get_requests[req_id]
        connector_get_key = f"{req_id[0:25]}_{target_stage_id}_{chunk_id}"

        # Use timeout=0 for non-blocking poll
        payload_data, size = self.connector.get(
            str(target_stage_id),
            str(stage_id),
            connector_get_key,
        )

        if payload_data:
            logger.debug(f"[Stage-{stage_id}] Received payload {payload_data}")

            self.request_prompt_token_ids[req_id] = payload_data.get("thinker_input_ids", [])
            # Update connector state
            self.get_requests[req_id] += 1
            req = self._pending_load_reqs[req_id]
            
            if stage_id != 2:
                req.additional_information = payload_data
                if payload_data.get("finished"):
                    self.finished_requests.add(req_id)
            else:
                if payload_data.get("finished"):
                    self.finished_requests.add(req_id)
                    req.status = RequestStatus.FINISHED_STOPPED

                # TODO: remove special handling for prompt token ids ?
                if chunk_id == 0:
                    req.prompt_token_ids = payload_data.get("code_predictor_codes", [])
                else:
                    req.prompt_token_ids += payload_data.get("code_predictor_codes", [])

            # Mark as finished for consumption
            with self.lock:
                self._finished_load_reqs.add(req_id)
                if req_id in self._pending_load_reqs:
                    del self._pending_load_reqs[req_id]
            logger.info(f"[Stage-{stage_id}] Received one chunk for key {connector_get_key}")

    def _process_single_save(self, task: dict):
        connector_put_key = task["put_key"]
        stage_id = task["stage_id"]
        next_stage_id = task["next_stage_id"]
        payload_data = task["data"]
        request_id = task["request_id"]

        success, size, metadata = self.connector.put(
            from_stage=str(stage_id),
            to_stage=str(next_stage_id),
            put_key=connector_put_key,
            data=payload_data,
        )

        if success:
            logger.info(f"[Stage-{stage_id}] Sent {connector_put_key}")
            with self.lock:
                self._finished_save_reqs.add(request_id)

        self.waiting_for_chunk_waiting_requests: deque[Any] = deque()
        self.waiting_for_chunk_running_requests: deque[Any] = deque()
        self.requests_with_ready_chunks = set()

    def process_pending_chunks(
        self,
        waiting_queue: Any,
        running_queue: list[Request],
    ) -> int:
        """
        Process pending chunks for waiting and running queues.
        Returns the number of running requests waiting for chunks.
        """
        self._process_chunk_queue(waiting_queue, self.waiting_for_chunk_waiting_requests, RequestStatus.WAITING)
        self._process_chunk_queue(
            running_queue,
            self.waiting_for_chunk_running_requests,
            RequestStatus.RUNNING,
        )
        return len(self.waiting_for_chunk_running_requests)

    def restore_queues(self, waiting_queue: Any, running_queue: list[Request]) -> None:
        """
        Restore requests waiting for chunk to the waiting and running queues.
        """
        # Add request waiting for chunk to the waiting and running queue
        for request in self.waiting_for_chunk_waiting_requests:
            waiting_queue.add_request(request)
        self.waiting_for_chunk_waiting_requests = deque()

        if self.waiting_for_chunk_running_requests:
            running_queue.extend(self.waiting_for_chunk_running_requests)
        self.waiting_for_chunk_running_requests = deque()

    def filter_scheduler_output(self, scheduler_output: Any) -> None:
        """
        Clean up ready chunks from scheduler output.
        """
        self._clear_chunk_ready(scheduler_output)

    def _process_chunk_queue(
        self,
        queue: Any,
        waiting_for_chunk_list: deque[Any],
        target_status: RequestStatus,
    ) -> None:
        queue_snapshot = list(queue)
        for request in queue_snapshot:
            if request.status != RequestStatus.WAITING_FOR_CHUNK:
                if request.request_id in self.requests_with_ready_chunks:
                    continue
                # Access finished_requests from self instead of connector
                if request.request_id in self.finished_requests:
                    self.finished_requests.remove(request.request_id)
                    request.additional_information = None
                    continue
                self.request_chunk(request)
                request.status = RequestStatus.WAITING_FOR_CHUNK
            else:
                finished_load_chunk_reqs = self.get_finished_load_requests()
                if request.request_id in finished_load_chunk_reqs:
                    request.status = target_status
                    self.requests_with_ready_chunks.add(request.request_id)
                    continue
            queue.remove(request)
            waiting_for_chunk_list.append(request)

    def _clear_chunk_ready(self, scheduler_output: Any) -> None:
        if scheduler_output.scheduled_new_reqs:
            for req_data in scheduler_output.scheduled_new_reqs:
                if req_data.req_id in self.requests_with_ready_chunks:
                    self.requests_with_ready_chunks.remove(req_data.req_id)

        if scheduler_output.scheduled_cached_reqs:
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                if req_id in self.requests_with_ready_chunks:
                    self.requests_with_ready_chunks.remove(req_id)

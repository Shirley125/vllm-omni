# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict, deque
from typing import Any

import torch
from vllm.v1.request import Request, RequestStatus

from .base import BasicOmniTransferManager
from ..utils.logging import get_connector_logger

logger = get_connector_logger(__name__)


class OmniChunkTransferManager(BasicOmniTransferManager):
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
        self.request_ids_mapping: dict[str, str] = {}

        self.waiting_for_chunk_waiting_requests: deque[Any] = deque()
        self.waiting_for_chunk_running_requests: deque[Any] = deque()
        self.requests_with_ready_chunks = set()

    def load(self, request):
        """Request to retrieve a chunk of data for a specific request.

        Args:
            request: The request object needing data.
        """
        stage_id = self.connector.stage_id
        request_id = request.request_id
        self.request_ids_mapping[request_id] = request.external_req_id

        if stage_id == 0:
            return
        if not hasattr(request, "additional_information"):
            request.additional_information = None
        with self.lock:
            self._pending_load_reqs[request_id] = request

    def save(self, pooling_output, request, custom_process_input_func=None):
        """Submit a chunk of data to be stored/sent asynchronously.

        Args:
            pooling_output: Partial pooling output dictionary
            request: Request object
            custom_process_input_func: Optional processing function
        """
        stage_id = self.connector.stage_id
        next_stage_id = stage_id + 1
        request_id = request.request_id

        # Snapshot prompt_token_ids
        prompt_token_ids = list(request.prompt_token_ids)
        self.request_prompt_token_ids[request_id] = prompt_token_ids
        chunk_id = self.put_requests[request_id]

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
                if self.request_payload.get(request_id) is None:
                    if not payload_data.get("finished"):
                        self.request_payload[request_id] = payload_data
                        return
                else:
                    save_payload = self.request_payload.pop(request_id)
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
                self.code_prompt_token_ids[request_id].append(payload_data.get("code_predictor_codes", []))
                length = len(self.code_prompt_token_ids[request_id])
                chunk_length = length % chunk_size
                if chunk_length != 0 and not payload_data.get("finished"):
                    return

                context_length = chunk_length if chunk_length != 0 else chunk_size
                end_index = min(length, left_context_size + context_length)
                payload_data["code_predictor_codes"] = (
                    torch.tensor(self.code_prompt_token_ids[request_id][-end_index:])
                        .transpose(0, 1)
                        .reshape(-1)
                        .tolist()
                )

        # Increment chunk_id here since we are committing to send
        self.put_requests[request_id] += 1
        connector_put_key = f"{request.external_req_id}_{stage_id}_{chunk_id}"

        task = {
            "stage_id": stage_id,
            "next_stage_id": next_stage_id,
            "put_key": connector_put_key,
            "data": payload_data,
            "request_id": request_id,
        }

        with self.lock:
            if request_id not in self._pending_save_reqs:
                self._pending_save_reqs[request_id] = deque()
            self._pending_save_reqs[request_id].append(task)

    def _process_single_recv(self, req_id: str):
        stage_id = self.connector.stage_id
        target_stage_id = stage_id - 1
        chunk_id = self.get_requests[req_id]
        external_req_id = self.request_ids_mapping.get(req_id, req_id)
        connector_get_key = f"{external_req_id}_{target_stage_id}_{chunk_id}"

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

    ########################################################################
    # Schedule Helper
    ########################################################################

    def process_pending_chunks(
            self,
            waiting_queue: Any,
            running_queue: list[Request],
    ) -> int:
        """
        Process pending chunks for waiting and running queues.
        Returns the number of running requests waiting for chunks.
        """
        if self.connector.stage_id != 0:
            return 0
        
        finished_reqs = self.get_finished_requests()
        self._process_chunk_queue(
            waiting_queue, 
            self.waiting_for_chunk_waiting_requests, 
            RequestStatus.WAITING, 
            finished_reqs
        )
        self._process_chunk_queue(
            running_queue,
            self.waiting_for_chunk_running_requests,
            RequestStatus.RUNNING,
            finished_reqs
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
            finished_reqs: set[str],
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
                self.load(request)
                request.status = RequestStatus.WAITING_FOR_CHUNK
            else:
                if request.request_id in finished_reqs:
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

    def get_finished_requests(self):
        with self.lock:
            finished_load = set(self._finished_load_reqs)
            self._finished_load_reqs = set()
        return finished_load

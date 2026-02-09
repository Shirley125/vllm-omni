# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict, deque
from typing import Any

import torch
from vllm.v1.request import Request, RequestStatus

from ..factory import OmniConnectorFactory
from ..utils.config import ConnectorSpec
from ..utils.logging import get_connector_logger
from .base import OmniModelMode, OmniTransferManagerBase

logger = get_connector_logger(__name__)


class OmniChunkTransferManager(OmniTransferManagerBase):
    """Manages asynchronous retrieval and storage of data chunks via OmniConnector."""

    def __init__(self, model_config: Any, mode: OmniModelMode):
        self.connector = self.create_connector(model_config)
        self.model_mode = mode
        super().__init__(model_config)

        # State specific to chunk management.
        # Next chunk index to send per external request id.
        self.put_req_chunk: dict[str, int] = defaultdict(int)
        # Next chunk index to receive per internal request id.
        self.get_req_chunk: dict[str, int] = defaultdict(int)
        # Requests that have received a "finished" signal from upstream.
        self.finished_requests: set[str] = set()
        # Cached payload per request for incremental concatenation.
        self.request_payload = {}
        # Latest prompt tokens received per request.
        self.request_prompt_token_ids: dict[str, list[int]] = defaultdict(list)
        # Code predictor prompt tokens (generation mode) per request.
        self.code_prompt_token_ids: dict[str, list[list[int]]] = defaultdict(list)
        # Map internal request id -> external request id used in connector keys.
        self.request_ids_mapping: dict[str, str] = {}

        # Temporary queues for requests waiting on chunks.
        self.waiting_for_chunk_waiting_requests: deque[Any] = deque()
        self.waiting_for_chunk_running_requests: deque[Any] = deque()
        # Requests that already have chunks ready for the scheduler.
        self.requests_with_ready_chunks = set()

    @classmethod
    def create_connector(cls, model_config: Any):
        connector_config = getattr(model_config, "stage_connector_config", None)
        if connector_config is None:
            connector_config = {}
        elif not isinstance(connector_config, dict):
            connector_config = {
                "name": getattr(connector_config, "name", None),
                "extra": getattr(connector_config, "extra", {}),
            }

        connector_specs = ConnectorSpec(
            name=connector_config.get("name", "SharedMemoryConnector"),
            extra=connector_config.get("extra", {}),
        )
        return OmniConnectorFactory.create_connector(connector_specs)

    def load(self, request):
        """Request to retrieve a chunk of data for a specific request.

        Args:
            request: The request object needing data.
        """
        stage_id = self.connector.stage_id
        request_id = request.request_id
        self.request_ids_mapping[request_id] = request.external_req_id

        # Stage 0 is the producer and does not pull from upstream.
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
        # Use external request id for connector keys across stages.
        external_req_id = request.external_req_id
        chunk_id = self.put_req_chunk[external_req_id]

        # Process payload in main thread to avoid race conditions on request state
        payload_data = None
        if custom_process_input_func:
            try:
                payload_data = custom_process_input_func(
                    transfer_manager=self,
                    pooling_output=pooling_output,
                    request=request,
                )

            except Exception as e:
                logger.error(f"Failed to use custom_process_input_func for payload extraction: {e}")

            if not payload_data:
                return

        # Increment chunk_id here since we are committing to send
        self.put_req_chunk[external_req_id] += 1
        connector_put_key = f"{external_req_id}_{stage_id}_{chunk_id}"

        task = {
            "stage_id": stage_id,
            "next_stage_id": next_stage_id,
            "put_key": connector_put_key,
            "data": payload_data,
            # Used only for logging when the background save fails.
            "request_id": external_req_id,
        }

        with self.lock:
            if external_req_id not in self._pending_save_reqs:
                self._pending_save_reqs[external_req_id] = deque()
            self._pending_save_reqs[external_req_id].append(task)

    def _process_single_recv(self, internal_req_id: str):
        # Poll upstream stage for the next chunk for this internal request id.
        current_stage_id = self.connector.stage_id
        upstream_stage_id = current_stage_id - 1
        expected_chunk_id = self.get_req_chunk[internal_req_id]
        external_req_id = self.request_ids_mapping.get(internal_req_id, internal_req_id)
        connector_get_key = f"{external_req_id}_{upstream_stage_id}_{expected_chunk_id}"

        # Use timeout=0 for non-blocking poll
        payload_data, size = self.connector.get(
            str(upstream_stage_id),
            str(current_stage_id),
            connector_get_key,
        )

        if payload_data:
            self.request_prompt_token_ids[internal_req_id] = payload_data.get("thinker_input_ids", [])
            # Update connector state
            self.get_req_chunk[internal_req_id] += 1
            req = self._pending_load_reqs[internal_req_id]

            if self.model_mode == OmniModelMode.MODE_AR:
                self._update_request_payload(external_req_id, payload_data)
                req.additional_information = payload_data
                if payload_data.get("finished"):
                    self.finished_requests.add(internal_req_id)
            else:
                if payload_data.get("finished"):
                    self.finished_requests.add(internal_req_id)
                    req.status = RequestStatus.FINISHED_STOPPED

                req.prompt_token_ids = payload_data.get("code_predictor_codes", [])
                req.num_computed_tokens = 0

            # Mark as finished for consumption
            with self.lock:
                self._finished_load_reqs.add(internal_req_id)
                if internal_req_id in self._pending_load_reqs:
                    del self._pending_load_reqs[internal_req_id]
            logger.debug(
                f"[Stage-{current_stage_id}] Received one chunk for key {connector_get_key}"
            )

    def _update_request_payload(self, req_id: str, payload_data: dict[str, Any]) -> dict[str, Any]:
        """Update the payload data for a request in the connector.

        Args:
            connector: OmniConnectorBase instance
            req_id: Request ID to update
            payload_data: New payload data to store
        """
        if req_id not in self.request_payload:
            self.request_payload[req_id] = payload_data
            return
        origin_payload = self.request_payload[req_id]
        for key, value in payload_data.items():
            if key == "finished":
                continue
            elif isinstance(value, torch.Tensor) and key in origin_payload:
                payload_data[key] = torch.cat([origin_payload[key], value], dim=0)
            elif isinstance(value, list) and key in origin_payload:
                payload_data[key] = origin_payload[key] + value

        self.request_payload[req_id] = payload_data
        return payload_data

    def _process_single_save(self, save_task: dict):
        # Push the prepared payload to the downstream stage.
        connector_put_key = save_task["put_key"]
        from_stage_id = save_task["stage_id"]
        to_stage_id = save_task["next_stage_id"]
        payload_data = save_task["data"]

        success, size, metadata = self.connector.put(
            from_stage=str(from_stage_id),
            to_stage=str(to_stage_id),
            put_key=connector_put_key,
            data=payload_data,
        )

        if success:
            logger.info(f"[Stage-{from_stage_id}] Sent {connector_put_key}")

    ########################################################################
    # Schedule Helper
    ########################################################################

    def process_pending_chunks(
        self,
        waiting_queue: Any,
        running_queue: list[Request],
    ) -> None:
        """
        Process pending chunks for waiting and running queues.
        """
        # No-op for stage 0, since it does not receive from upstream.
        if self.connector.stage_id == 0:
            return
        finished_load_reqs = self.get_finished_requests()
        self._process_chunk_queue(
            waiting_queue, self.waiting_for_chunk_waiting_requests, RequestStatus.WAITING, finished_load_reqs
        )
        self._process_chunk_queue(
            running_queue, self.waiting_for_chunk_running_requests, RequestStatus.RUNNING, finished_load_reqs
        )

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

    def filter_scheduler_output(
        self,
        scheduler_output: Any,
        requests: dict[str, Request] | None = None,
    ) -> None:
        """
        Add additional info for cached requests and
        clean up ready chunks from scheduler output.
        """
        if requests is not None:
            self.attach_cached_additional_information(scheduler_output, requests)
        self._clear_chunk_ready(scheduler_output)

    @staticmethod
    def attach_cached_additional_information(scheduler_output: Any, requests: dict[str, Request]) -> None:
        cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if not cached_reqs:
            return
        if not hasattr(cached_reqs, "additional_information"):
            cached_reqs.additional_information = {}
        for req_id in cached_reqs.req_ids:
            request = requests.get(req_id) if req_id else None
            additional_info = getattr(request, "additional_information", None) if request else None
            cached_reqs.additional_information[req_id] = additional_info

    def _process_chunk_queue(
        self,
        queue: Any,
        waiting_for_chunk_list: deque[Any],
        target_status: RequestStatus,
        finished_load_reqs: set[str],
    ) -> None:
        queue_snapshot = list(queue)
        for request in queue_snapshot:
            if request.status != RequestStatus.WAITING_FOR_CHUNK:
                if request.request_id in self.requests_with_ready_chunks:
                    # Requests that have loaded chunk from last round
                    # of schedule, but have not scheduled
                    continue
                if request.request_id in self.finished_requests:
                    request.additional_information = {}
                    continue
                # Requests that waiting for chunk
                self.load(request)
                request.status = RequestStatus.WAITING_FOR_CHUNK
            else:
                if request.request_id in finished_load_reqs:
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

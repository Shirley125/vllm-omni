# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict, deque
from typing import Any

import torch
from vllm.v1.request import Request, RequestStatus

from ..utils.config import ConnectorSpec
from ..utils.logging import get_connector_logger
from .base import OmniTransferManagerBase

logger = get_connector_logger(__name__)


class OmniChunkTransferManager(OmniTransferManagerBase):
    """Manages asynchronous retrieval and storage of data chunks via OmniConnector."""

    @classmethod
    def maybe_create(
        cls,
        config: Any | None = None,
        *,
        connector: Any | None = None,
        enabled: bool | None = None,
        connector_spec: ConnectorSpec | dict[str, Any] | None = None,
        connector_config: dict[str, Any] | None = None,
    ) -> "OmniChunkTransferManager | None":
        """Create a chunk manager when async chunking is enabled."""
        if connector is not None:
            return cls(connector)

        if not cls._is_async_chunk_enabled(config, enabled):
            return None

        built_connector = cls.build_connector(
            config=config,
            connector_spec=connector_spec,
            connector_config=connector_config,
        )
        if built_connector is None:
            return None
        return cls(built_connector)

    @staticmethod
    def _is_async_chunk_enabled(config: Any | None, enabled: bool | None) -> bool:
        if enabled is not None:
            return bool(enabled)
        if config is None:
            return False
        if isinstance(config, dict):
            return bool(config.get("async_chunk", False))
        return bool(getattr(config, "async_chunk", False))

    @classmethod
    def build_connector(
        cls,
        *,
        connector: Any | None = None,
        config: Any | None = None,
        connector_spec: ConnectorSpec | dict[str, Any] | None = None,
        connector_config: dict[str, Any] | None = None,
    ):
        """Build an OmniConnector using chunk-specific config resolution."""
        if connector is not None:
            return connector

        spec = cls._resolve_connector_spec(
            config=config, connector_spec=connector_spec, connector_config=connector_config
        )
        if spec is None:
            return None
        return super().build_connector(connector_spec=spec)

    @classmethod
    def _resolve_connector_spec(
        cls,
        *,
        config: Any | None = None,
        connector_spec: ConnectorSpec | dict[str, Any] | None = None,
        connector_config: dict[str, Any] | None = None,
    ) -> ConnectorSpec | None:
        source = cls._select_spec_source(config, connector_spec, connector_config)
        spec = cls._spec_from_source(source)
        if spec is None:
            return None

        stage_id = cls._extract_stage_id(config)
        if stage_id is not None and "stage_id" not in (spec.extra or {}):
            return ConnectorSpec(name=spec.name, extra={**(spec.extra or {}), "stage_id": stage_id})
        return spec

    @staticmethod
    def _select_spec_source(
        config: Any | None,
        connector_spec: ConnectorSpec | dict[str, Any] | None,
        connector_config: dict[str, Any] | None,
    ) -> Any | None:
        if connector_spec is not None:
            return connector_spec
        if connector_config is not None:
            return connector_config
        if config is None:
            return None

        if isinstance(config, dict):
            for key in (
                "stage_connector_config",
                "connector_config",
                "omni_connector_config",
                "stage_connector_spec",
                "connector_spec",
            ):
                if key in config and config[key]:
                    return config[key]
            if "spec" in config and isinstance(config.get("spec"), dict):
                return config["spec"]
            if "name" in config or "type" in config:
                return config
            return None

        for attr in (
            "stage_connector_config",
            "connector_config",
            "omni_connector_config",
            "stage_connector_spec",
            "connector_spec",
        ):
            value = getattr(config, attr, None)
            if value:
                return value
        return None

    @staticmethod
    def _spec_from_source(source: Any | None) -> ConnectorSpec | None:
        if source is None:
            return None
        if isinstance(source, ConnectorSpec):
            return source
        if not isinstance(source, dict):
            return None

        spec_dict = source.get("spec") if isinstance(source.get("spec"), dict) else source
        if isinstance(spec_dict, ConnectorSpec):
            return spec_dict

        if "name" in spec_dict:
            name = spec_dict.get("name")
            extra = spec_dict.get("extra", {}) or {}
        elif "type" in spec_dict:
            name = spec_dict.get("type")
            extra = {}
            raw_extra = spec_dict.get("extra")
            if isinstance(raw_extra, dict):
                extra.update(raw_extra)
            for key, value in spec_dict.items():
                if key in ("type", "name", "extra", "spec"):
                    continue
                if key not in extra:
                    extra[key] = value
        else:
            return None

        if not name:
            return None
        return ConnectorSpec(name=name, extra=extra or {})

    @staticmethod
    def _extract_stage_id(config: Any | None) -> Any | None:
        if config is None:
            return None
        if isinstance(config, dict):
            return config.get("stage_id")
        return getattr(config, "stage_id", None)

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
        request_id = request.external_req_id
        chunk_id = self.put_requests[request_id]

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
        self.put_requests[request_id] += 1
        connector_put_key = f"{request_id}_{stage_id}_{chunk_id}"

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
            self.request_prompt_token_ids[req_id] = payload_data.get("thinker_input_ids", [])
            # Update connector state
            self.get_requests[req_id] += 1
            req = self._pending_load_reqs[req_id]

            if stage_id != 2:
                self._update_request_payload(external_req_id, payload_data)
                req.additional_information = payload_data
                if payload_data.get("finished"):
                    self.finished_requests.add(req_id)
            else:
                if payload_data.get("finished"):
                    self.finished_requests.add(req_id)
                    req.status = RequestStatus.FINISHED_STOPPED

                req.prompt_token_ids = payload_data.get("code_predictor_codes", [])
                req.num_computed_tokens = 0

            # Mark as finished for consumption
            with self.lock:
                self._finished_load_reqs.add(req_id)
                if req_id in self._pending_load_reqs:
                    del self._pending_load_reqs[req_id]
            logger.info(f"[Stage-{stage_id}] Received one chunk for key {connector_get_key}")

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

    def _process_single_save(self, task: dict):
        connector_put_key = task["put_key"]
        stage_id = task["stage_id"]
        next_stage_id = task["next_stage_id"]
        payload_data = task["data"]

        success, size, metadata = self.connector.put(
            from_stage=str(stage_id),
            to_stage=str(next_stage_id),
            put_key=connector_put_key,
            data=payload_data,
        )

        if success:
            logger.info(f"[Stage-{stage_id}] Sent {connector_put_key}")

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
        if self.connector.stage_id == 0:
            return 0
        finished_reqs = self.get_finished_requests()
        self._process_chunk_queue(
            waiting_queue, self.waiting_for_chunk_waiting_requests, RequestStatus.WAITING, finished_reqs
        )
        self._process_chunk_queue(
            running_queue, self.waiting_for_chunk_running_requests, RequestStatus.RUNNING, finished_reqs
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
                    request.additional_information = {}
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

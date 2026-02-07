# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time
from typing import Any

from ..factory import OmniConnectorFactory
from ..utils.config import ConnectorSpec
from ..utils.logging import get_connector_logger

logger = get_connector_logger(__name__)


class OmniTransferManagerBase:
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

    @classmethod
    def build_connector(
        cls,
        *,
        connector: Any | None = None,
        config: Any | None = None,
        connector_spec: ConnectorSpec | dict[str, Any] | None = None,
        connector_config: dict[str, Any] | None = None,
    ):
        """Build an OmniConnector instance from flexible config inputs."""
        if connector is not None:
            return connector

        spec = cls._resolve_connector_spec(
            config=config, connector_spec=connector_spec, connector_config=connector_config
        )
        if spec is None:
            return None
        return OmniConnectorFactory.create_connector(spec)

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

    def load(self, *args, **kwargs):
        """Register a request to load data. To be implemented by subclasses."""
        raise NotImplementedError

    def save(self, *args, **kwargs):
        """Submit data to be saved/sent. To be implemented by subclasses."""
        raise NotImplementedError

    def get_finished_requests(self):
        raise NotImplementedError

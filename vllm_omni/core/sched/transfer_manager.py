from __future__ import annotations

from typing import Any, Mapping, TYPE_CHECKING

from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


class OmniTransferManagerBase:
    """Base helper for building omni connectors from vLLM config."""

    def __init__(self, vllm_config: Any) -> None:
        self.vllm_config = vllm_config
        self.model_config = getattr(vllm_config, "model_config", None)
        self.omni_connector = None
        if self.model_config is not None and getattr(self.model_config, "async_chunk", False):
            self.omni_connector = self._create_connector(self.model_config)

    @classmethod
    def maybe_create(cls, vllm_config: Any) -> "OmniTransferManagerBase | None":
        model_config = getattr(vllm_config, "model_config", None) if vllm_config is not None else None
        if model_config is None or not getattr(model_config, "async_chunk", False):
            return None
        return cls(vllm_config)

    @staticmethod
    def _create_connector(model_config: Any):
        connector_config = getattr(model_config, "stage_connector_config", None)
        if not isinstance(connector_config, dict):
            connector_config = {}
        connector_specs = ConnectorSpec(
            name=connector_config.get("name", "SharedMemoryConnector"),
            extra=connector_config.get("extra", {}),
        )
        return OmniConnectorFactory.create_connector(connector_specs)


class OmniChunkTransferManager(OmniTransferManagerBase):
    """Chunk-level transfer helper for scheduler outputs."""

    def filter_scheduler_output(
        self,
        scheduler_output: "SchedulerOutput",
        requests: Mapping[str, "Request"] | None = None,
    ) -> None:
        cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached_reqs is None:
            return

        existing = getattr(cached_reqs, "additional_information", None)
        if not isinstance(existing, dict):
            cached_reqs.additional_information = {}

        req_map = requests or {}
        for req_id in getattr(cached_reqs, "req_ids", []):
            request = req_map.get(req_id) if req_id else None
            additional_info = getattr(request, "additional_information", None) if request else None
            cached_reqs.additional_information[req_id] = additional_info

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm_omni.distributed.omni_connectors.connectors.base import OmniConnectorBase
from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory

_OMNI_CONNECTOR_AGENT: OmniConnectorBase | None = None


def get_omni_transfer() -> OmniConnectorBase:
    assert _OMNI_CONNECTOR_AGENT is not None, (
        "disaggregated EC cache is not initialized")
    return _OMNI_CONNECTOR_AGENT


def has_omni_transfer() -> bool:
    return _OMNI_CONNECTOR_AGENT is not None


def ensure_omni_transfer_initialized(stage_id: int, device) -> None:
    """
    Initialize EC cache connector.
    """

    global _OMNI_CONNECTOR_AGENT

    if  _OMNI_CONNECTOR_AGENT is None:
        # _OMNI_CONNECTOR_AGENT = OmniConnectorFactory.create_connector(
        #     spec=vllm_config.connector_spec)
        from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec
        extra = {"shm_threshold_bytes": 65536, "stage_id": stage_id, "device": device}
        connector_spec = ConnectorSpec(
            name="SharedMemoryConnector",
            extra=extra,
        )
        _OMNI_CONNECTOR_AGENT = OmniConnectorFactory.create_connector(
            spec=connector_spec)

        

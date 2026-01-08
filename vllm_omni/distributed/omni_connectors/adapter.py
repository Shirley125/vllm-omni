# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# temporary for compatibility with vllm_omni.entrypoints.omni_stage.py
# and vllm_omni.entrypoints.omni_llm.py

import time
from collections.abc import Callable
from typing import Any

from vllm_omni.entrypoints.stage_utils import OmniStageTaskType

from .utils.logging import get_connector_logger

logger = get_connector_logger(__name__)


def try_send_via_connector(
    connector: Any,
    stage_id: int,
    next_stage_id: int,
    req_id: str,
    next_inputs: Any,
    sampling_params: Any,
    original_prompt: Any,
    next_stage_queue_submit_fn: Callable[[dict[str, Any]], None],
    metrics: Any,
) -> bool:
    """
    Attempts to send data via OmniConnector.
    Returns True if successful, False otherwise.
    Encapsulates the logic of preparing payload, sending via connector,
    sending notification, and recording metrics.
    """
    try:
        t0 = time.time()

        # Prepare data for connector
        payload_data = {
            "engine_inputs": next_inputs,
            "sampling_params": sampling_params,
            "metadata": {
                "original_prompt": original_prompt,
                "stage_transition": f"{stage_id}->{next_stage_id}",
                "timestamp": time.time(),
            },
        }

        # Send data via connector
        success, serialized_size, metadata = connector.put(str(stage_id), str(next_stage_id), str(req_id), payload_data)

        if success:
            # Send lightweight notification via queue
            notify_payload = {
                "type": OmniStageTaskType.GENERATE,
                "request_id": req_id,
                "sampling_params": sampling_params,
                "from_connector": True,
                "from_stage": str(stage_id),
                "to_stage": str(next_stage_id),
                "sent_ts": time.time(),
            }
            # Merge connector metadata (e.g. shm handle or inline data) into queue payload
            if metadata:
                notify_payload["connector_metadata"] = metadata

            next_stage_queue_submit_fn(notify_payload)

            t1 = time.time()
            tx_ms = (t1 - t0) * 1000.0

            metrics.on_forward(
                stage_id,
                next_stage_id,
                req_id,
                serialized_size,  # Use size from connector
                float(tx_ms),
                True,  # Mark as using connector
            )
            return True
        else:
            # If put returned False, we let the caller handle fallback
            return False

    except Exception as e:
        logger.warning(
            "[Orchestrator] OmniConnector failed for req %s: %s; falling back to queue",
            req_id,
            e,
        )
        return False


def try_recv_via_connector(
    task: dict[str, Any],
    connectors: dict[Any, Any],
    stage_id: int,
) -> tuple[Any, dict[str, Any] | None]:
    """
    Attempts to resolve input data from either connector or IPC.
    Returns (engine_inputs, rx_metrics) or (None, None) if failed/skipped.
    """
    rid = task["request_id"]

    if task.get("from_connector"):
        from_stage = task.get("from_stage")
        to_stage = str(stage_id)

        if not from_stage:
            logger.error(
                "[Stage-%s] 'from_connector' is true but 'from_stage' is missing for request %s", stage_id, rid
            )
            return None, None

        # Get connector for this edge
        connector_key = (from_stage, to_stage)
        connector = connectors.get(connector_key)

        if connector:
            try:
                # Get data from connector with timeout
                _t_start = time.time()
                connector_metadata = task.get("connector_metadata")
                payload = connector.get(from_stage, to_stage, str(rid), metadata=connector_metadata)
                _t_end = time.time()

                if payload:
                    if isinstance(payload, tuple):
                        payload_data, serialized_size = payload
                    else:
                        payload_data = payload
                        serialized_size = len(connector.serialize_obj(payload_data))
                else:
                    payload_data = None
                    serialized_size = 0

                if payload_data and isinstance(payload_data, dict):
                    ein = payload_data.get("engine_inputs")
                    decode_ms = (_t_end - _t_start) * 1000.0

                    rx_metrics = {"rx_decode_time_ms": decode_ms, "rx_transfer_bytes": serialized_size}
                    return ein, rx_metrics
                else:
                    logger.error(
                        "[Stage-%s] Failed to get data from connector for request %s or payload is empty", stage_id, rid
                    )
                    return None, None
            except Exception as e:
                logger.error("[Stage-%s] Error retrieving data from connector for request %s: %s", stage_id, rid, e)
                return None, None
        else:
            logger.error(
                "[Stage-%s] No connector found for edge %s -> %s for request %s", stage_id, from_stage, to_stage, rid
            )
            return None, None
    else:
        # Data comes from queue as usual (e.g. seed request for Stage-0)
        # Since fallback logic is deprecated, we assume this is a direct inputs payload.
        # We still need to decode it if it used SHM (via legacy stage_utils logic, or new shm_connector format)
        # For Stage-0 specifically, 'engine_inputs' is often directly in the task dict.

        # Try to use the new stage_utils which uses OmniSerializer
        from vllm_omni.entrypoints.stage_utils import maybe_load_from_ipc_with_metrics

        try:
            ein, metrics = maybe_load_from_ipc_with_metrics(task, "engine_inputs", "engine_inputs_shm")
            # If metrics are empty or zero, we might want to populate dummy metrics
            return ein, metrics
        except Exception:
            # If engine_inputs is missing, it might be a different kind of payload,
            # but for Stage-0 seed it should be there.
            # We'll return None to let caller handle error if strictly required.
            return None, None


def get_chunk(connector, scheduler_output):
    """Retrieve a chunk of pooling output.

    Args:
        scheduler_output: Partial scheduler output dictionary

    Returns:
        dict[str, Any] | None: Pooling output dictionary or None if not found
    """
    stage_id = connector.stage_id
    if stage_id == 0:
        return

    target_stage_id = stage_id - 1
    # Handle new requests
    for new_req_data in scheduler_output.scheduled_new_reqs:
        req_id = new_req_data.req_id
        chunk_id = connector.get_requests[req_id]
        connector_get_key = f"{req_id}_{target_stage_id}_{chunk_id}"
        payload_data = get_through_connector(connector, target_stage_id, stage_id, req_id, connector_get_key)
        if payload_data:
            new_req_data.additional_information = payload_data
            # self.additional_information[req_id] = payload_data

    # Handle cached/running requests
    cached_reqs = scheduler_output.scheduled_cached_reqs
    if not hasattr(cached_reqs, "additional_information"):
        cached_reqs.additional_information = {}

    for i, req_id in enumerate(cached_reqs.req_ids):
        if req_id in connector.finished_requests:
            continue
        chunk_id = connector.get_requests[req_id]
        connector_get_key = f"{req_id}_{target_stage_id}_{chunk_id}"
        payload_data = get_through_connector(connector, target_stage_id, stage_id, req_id, connector_get_key)
        if payload_data:
            cached_reqs.additional_information[req_id] = payload_data


def get_through_connector(connector, target_stage_id, stage_id, req_id, connector_get_key):
    # Wait for data from previous stage
    import time

    # TODO: add correct check mechanism for the payload_data
    max_wait = 300
    for _ in range(max_wait):
        result = connector.get(
            from_stage=str(target_stage_id),
            to_stage=str(stage_id),
            get_key=connector_get_key,
        )
        payload_data = None
        if result:
            payload_data, size = result
            if payload_data:
                # TODO: custom validate the payload_data ?
                if stage_id == 2:
                    if not validate_talker_output(payload_data):
                        logger.warning(
                            f"[Stage-{stage_id}] Received invalid token IDs for request {req_id}. Waiting..."
                        )
                        continue

                # logger.info(f"recved payload_data: {payload_data}")
                connector.request_prompt_token_ids[req_id] = payload_data.get("thinker_input_ids", [])
                connector.get_requests[req_id] += 1
                logger.info(f"[Stage-{stage_id}] Received one chunk for request {req_id}")
                break
        time.sleep(1)
    return payload_data


def validate_talker_output(payload_data):
    """Validate that we received proper token IDs"""
    talker_output_ids = payload_data.get("talker_output_ids", [])
    if talker_output_ids and len(talker_output_ids) > 0:
        token_count = len(talker_output_ids)
        if token_count % 16 == 0 or token_count > 100:
            return True
    return False


def get_chunk_for_generation(connector, request):
    """Retrieve a chunk of pooling output.

    Args:
        request: Request object

    Returns:
        dict[str, Any] | None: Pooling output dictionary or None if not found
    """
    stage_id = connector.stage_id
    target_stage_id = stage_id - 1
    request_id = request.request_id

    if request_id in connector.finished_requests:
        return

    chunk_id = connector.get_requests[request_id]
    connector_get_key = f"{request_id}_{target_stage_id}_{chunk_id}"
    payload_data = get_through_connector(connector, target_stage_id, stage_id, request_id, connector_get_key)
    if not payload_data:
        return

    if payload_data.get("finished"):
        connector.finished_requests.add(request_id)
    connector.get_requests[request_id] += 1
    # request.prompt_token_ids = payload_data.get("code_predictor_codes", []).tolist()
    logger.info(f"[Stage-{stage_id}] Received one chunk for request {connector_get_key}")

    if chunk_id == 0:
        request.prompt_token_ids = payload_data.get("code_predictor_codes", []).tolist()
    else:
        request.prompt_token_ids += payload_data.get("code_predictor_codes", []).tolist()


def put_chunk(connector, pooling_output, request, custom_process_input_func=None):
    """Store a chunk of pooling output.

    Args:
        connector: OmniConnectorBase instance
        pooling_output: Partial pooling output dictionary
        request: Request object
    """
    stage_id = connector.stage_id
    next_stage_id = stage_id + 1
    request_id = request.request_id
    prompt_token_ids = request.prompt_token_ids
    connector.request_prompt_token_ids[request_id] = prompt_token_ids
    chunk_id = connector.put_requests[request_id]
    connector_put_key = f"{request_id}_{stage_id}_{chunk_id}"
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

        success, size, metadata = connector.put(
            from_stage=str(stage_id), to_stage=str(next_stage_id), put_key=connector_put_key, data=payload_data
        )

        if success:
            connector.put_requests[request_id] += 1
            logger.info(f"[Stage-{stage_id}] Sent chunk {chunk_id} for request {connector_put_key}")

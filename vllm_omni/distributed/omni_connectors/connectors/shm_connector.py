# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any
import os
import time
import safetensors
import torch

from collections import defaultdict

from vllm.v1.request import RequestStatus

from collections import defaultdict
from vllm_omni.entrypoints.stage_utils import shm_read_bytes, shm_write_bytes

from ..utils.logging import get_connector_logger
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)


class SharedMemoryConnector(OmniConnectorBase):
    """
    Connector that uses SharedMemory for large objects and inline data for small objects.
    Acts as a unified replacement for the legacy IPC fallback logic.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.stage_id = config.get("stage_id", -1)
        self.device = config.get("device", "cuda:0")
        self.put_requests: dict[str, int] = defaultdict(int)
        self.get_requests: dict[str, int] = defaultdict(int)
        self.finished_requests: set[str] = set()
        self.request_prompt_token_ids: dict[str, list[int]] = defaultdict(list)
        self._storage_path = config.get("store_path", "/tmp")
        # Default threshold matches legacy behavior (64KB)
        self.threshold = int(config.get("shm_threshold_bytes", 65536))
        self._metrics = {
            "puts": 0,
            "gets": 0,
            "bytes_transferred": 0,
            "shm_writes": 0,
            "inline_writes": 0,
        }

    def put(
        self, from_stage: str, to_stage: str, request_id: str, data: Any
    ) -> tuple[bool, int, dict[str, Any] | None]:
        try:
            # Always serialize first to check size (and for SHM writing)
            # Note: For extremely large objects in "inline" mode (e.g. Ray),
            # we might double-serialize if we're not careful, but here we assume
            # if it's huge we use SHM, or if Ray, threshold is maxsize.
            payload = self.serialize_obj(data)
            size = len(payload)

            if size > self.threshold:
                # Use Shared Memory
                meta = shm_write_bytes(payload)
                # meta contains {'name': ..., 'size': ...}
                metadata = {"shm": meta, "size": size}
                self._metrics["shm_writes"] += 1
            else:
                # Inline - pass bytes directly to avoid double serialization of the object
                # We already serialized it to check size, so we pass the bytes.
                # The Queue will pickle these bytes (fast), avoiding re-serializing the complex object.
                metadata = {"inline_bytes": payload, "size": size}
                self._metrics["inline_writes"] += 1

            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += size

            return True, size, metadata

        except Exception as e:
            logger.error(f"SharedMemoryConnector put failed for req {request_id}: {e}")
            return False, 0, None

    def get(
        self, from_stage: str, to_stage: str, request_id: str, metadata: dict[str, Any] | None = None
    ) -> tuple[Any, int] | None:
        if not metadata:
            logger.error(f"SharedMemoryConnector get called without metadata for req {request_id}")
            return None

        try:
            obj = None
            size = 0

            if "shm" in metadata:
                meta = metadata["shm"]
                # shm_read_bytes handles reading and unlinking
                data_bytes = shm_read_bytes(meta)
                obj = self.deserialize_obj(data_bytes)
                size = metadata.get("size", len(data_bytes))
            elif "inline_bytes" in metadata:
                # Deserialize bytes back to object
                payload = metadata["inline_bytes"]
                obj = self.deserialize_obj(payload)
                size = metadata.get("size", len(payload))
            elif "inline" in metadata:
                obj = metadata["inline"]
                size = metadata.get("size", 0)
                if size == 0:
                    # Fallback if size wasn't recorded
                    try:
                        size = len(self.serialize_obj(obj))
                    except Exception:
                        pass
            else:
                logger.error(
                    f"Unknown metadata format in SharedMemoryConnector for req {request_id}: {list(metadata.keys())}"
                )
                return None

            self._metrics["gets"] += 1
            return obj, size

        except Exception as e:
            logger.error(f"SharedMemoryConnector get failed for req {request_id}: {e}")
            return None

    def cleanup(self, request_id: str) -> None:
        # SHM segments are automatically unlinked during 'get' (shm_read_bytes).
        # If 'get' is never called (e.g. error flow), the SHM segment might leak.
        # A robust implementation might track created segments and unlink them here
        # if they haven't been consumed.
        # For now, we rely on the consumer to read and unlink.
        pass

    def health(self) -> dict[str, Any]:
        return {"status": "healthy", "threshold": self.threshold, **self._metrics}

    def _found_match_for_stage_chunk(self, stage_chunk_key) -> bool:
        """Check if the cache is hit for the request."""
        filename = self._generate_filename_debug(stage_chunk_key)
        return os.path.exists(filename)

    def _generate_foldername_debug(
            self,
            stage_chunk_key: str,
            create_folder: bool = True,  # <- now defaults to True
    ) -> str:
        """
        Return the folder in which the cache for this stage_chunk_key lives.
        If `create_folder` is True (default) the directory is created
        recursively the first time it is needed.
        """
        foldername = os.path.join(self._storage_path, stage_chunk_key)
        if create_folder:
            os.makedirs(foldername, exist_ok=True)
        return foldername

    def _generate_filename_debug(self, stage_chunk_key: str) -> str:
        """
        Return the full path of the safetensors file for this stage_chunk_key.
        Ensures the parent directory exists because
        `_generate_foldername_debug` is called with its default
        (`create_folder=True`).
        """
        foldername = self._generate_foldername_debug(
            stage_chunk_key)  # <- folder auto-created
        return os.path.join(foldername, "stage_cache.safetensors")

    def put_chunk(self, pooling_output, request) -> None:
        """Store a chunk of pooling output.

        Args:
            pooling_output: Partial pooling output dictionary
            request: Request object
        """
        if self.stage_id == 0:
            return self.put_chunk_stage_0(pooling_output, request)
        return self.put_chunk_stage_1(pooling_output, request)

    def get_chunk(self, scheduler_output) -> dict[str, Any] | None:
        """Retrieve a chunk of pooling output.

        Args:
            scheduler_output: Partial scheduler output dictionary

        Returns:
            dict[str, Any] | None: Pooling output dictionary or None if not found
        """
        if self.stage_id == 0:
            return

        if self.stage_id == 1:
            return self.get_chunk_stage_0(scheduler_output)
        return self.get_chunk_stage_1(scheduler_output)

    def put_chunk_stage_0(self, pooling_output, request) -> None:
        """Store a chunk of pooling output at stage 1.

        Args:
            pooling_output: Partial pooling output dictionary
            request: Request object
        """
        request_id = request.request_id
        prompt_token_ids = request.prompt_token_ids
        self.request_prompt_token_ids[request_id] = prompt_token_ids
        token_ids = request.all_token_ids
        chunk = self.put_requests[request_id]
        stage_key = f"{request_id}_{self.stage_id}_{chunk}"
        filename = self._generate_filename_debug(stage_key)
        if os.path.exists(filename):
            return

        tensors = {
            "thinker_embeddings": pooling_output["0"].detach().cpu(),
            "thinker_hidden_states": pooling_output["24"].detach().cpu(),
            "thinker_input_ids": torch.tensor(prompt_token_ids, dtype=torch.int32),
            "thinker_sequences": torch.tensor(token_ids, dtype=torch.int32),
            "tts_bos_embed": pooling_output["tts_bos_embed"][0].detach().cpu(),
            "tts_eos_embed": pooling_output["tts_eos_embed"][0].detach().cpu(),
            "tts_pad_embed": pooling_output["tts_pad_embed"][0].detach().cpu(),
            "finished": torch.tensor(request.is_finished(), dtype=torch.bool),

        }
        self.put_requests[request_id] += 1
        safetensors.torch.save_file(tensors, filename)

    def put_chunk_stage_1(self, pooling_output, request) -> None:
        """Store a chunk of pooling output at stage 1.

        Args:
            pooling_output: Partial pooling output dictionary
            request: Request object

        Returns:
            dict[str, Any] | None: Pooling output dictionary or None if not found
        """
        if pooling_output.get("code_predictor_codes", None) is None:
            return

        request_id = request.request_id
        chunk = self.put_requests[request_id]
        stage_key = f"{request_id}_{self.stage_id}_{chunk}"
        filename = self._generate_filename_debug(stage_key)
        if os.path.exists(filename):
            return None
        code = torch.flatten(pooling_output["code_predictor_codes"]).detach().cpu()
        tensors = {
            "code_predictor_codes": code,
            "finished": torch.tensor(request.is_finished(), dtype=torch.bool),
        }
        self.put_requests[request_id] += 1
        safetensors.torch.save_file(tensors, filename)

    def get_chunk_stage_0(self, scheduler_output) -> dict[str, Any] | None:
        """Retrieve a chunk of pooling output at stage 1.

        Args:
            scheduler_output: Partial scheduler output dictionary

        Returns:
            dict[str, Any] | None: Pooling output dictionary or None if not found
        """

        target_stage_id = self.stage_id - 1
        for new_req_data in scheduler_output.scheduled_new_reqs:
            request_id = new_req_data.req_id
            chunk = self.get_requests[request_id]
            stage_key = f"{request_id}_{target_stage_id}_{chunk}"
            # TODO
            wait_time = 30
            for _ in range(wait_time):
                if self._found_match_for_stage_chunk(stage_key):
                    break
                else:
                    time.sleep(1)
            filename = self._generate_filename_debug(stage_key)
            output_data = safetensors.torch.load_file(filename)
            if output_data.get("finished"):
                self.finished_requests.add(request_id)
            tensors = {
                "thinker_embeddings": output_data.get("thinker_embeddings").detach(),
                "thinker_hidden_states": output_data.get("thinker_hidden_states").detach(),
                "thinker_input_ids": output_data.get("thinker_input_ids").tolist(),
                "thinker_sequences": output_data.get("thinker_sequences").tolist(),
                "tts_bos_embed": (output_data.get("tts_bos_embed").detach()),
                "tts_eos_embed": (output_data.get("tts_eos_embed").detach()),
                "tts_pad_embed": (output_data.get("tts_pad_embed").detach()),
            }
            self.get_requests[request_id] += 1
            self.request_prompt_token_ids[request_id] = tensors["thinker_input_ids"]
            new_req_data.additional_information = tensors
            logger.info(f"get chunk {stage_key} from shm connector, tensors:{tensors}")

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for _, request_id in enumerate(cached_reqs.req_ids):
            if request_id in self.finished_requests:
                continue
            chunk = self.get_requests[request_id]
            stage_key = f"{request_id}_{target_stage_id}_{chunk}"
            # TODO
            wait_time = 30
            for _ in range(wait_time):
                if self._found_match_for_stage_chunk(stage_key):
                    break
                else:
                    time.sleep(1)
            filename = self._generate_filename_debug(stage_key)
            output_data = safetensors.torch.load_file(filename)
            if output_data.get("finished"):
                self.finished_requests.add(request_id)
            prompt_token_ids = self.request_prompt_token_ids[request_id]
            tensors = {
                "thinker_embeddings": output_data.get("thinker_embeddings").detach(),
                "thinker_hidden_states": output_data.get("thinker_hidden_states").detach(),
                "thinker_sequences": output_data.get("thinker_sequences").tolist(),
                "thinker_input_ids": prompt_token_ids,
                "tts_bos_embed": (output_data.get("tts_bos_embed").detach()),
                "tts_eos_embed": (output_data.get("tts_eos_embed").detach()),
                "tts_pad_embed": (output_data.get("tts_pad_embed").detach()),
            }
            self.get_requests[request_id] += 1
            cached_reqs.additional_information = tensors
            logger.info(f"get chunk {stage_key} from shm connector, tensors:{tensors}")

    def get_chunk_stage_1(self, request) -> dict[str, Any] | None:
        """Retrieve a chunk of pooling output at stage 1.

        Args:
            scheduler_output: Partial scheduler output dictionary

        Returns:
            dict[str, Any] | None: Pooling output dictionary or None if not found
        """

        request_id = request.request_id
        if request_id in self.finished_requests:
            return
        target_stage_id = self.stage_id - 1
        chunk = self.get_requests[request_id]
        stage_key = f"{request_id}_{target_stage_id}_{chunk}"
        # TODO
        wait_time = 30
        for _ in range(wait_time):
            if self._found_match_for_stage_chunk(stage_key):
                break
            else:
                time.sleep(1)
        filename = self._generate_filename_debug(stage_key)
        output_data = safetensors.torch.load_file(filename)
        if output_data.get("finished"):
            self.finished_requests.add(request_id)
            request.status = RequestStatus.FINISHED_STOPPED
        self.get_requests[request_id] += 1
        if chunk == 0:
            request.prompt_token_ids = output_data["code_predictor_codes"].tolist()
        else:
            request.prompt_token_ids += output_data["code_predictor_codes"].tolist()
        logger.info(f"get chunk {stage_key} from shm connector, tensors:{request.prompt_token_ids}")


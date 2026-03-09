# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import fcntl
import os
import time
from multiprocessing import shared_memory as shm_pkg
from typing import Any

from vllm_omni.entrypoints.stage_utils import shm_read_bytes, shm_write_bytes

from ..utils.logging import get_connector_logger
from ..utils.perf_logging import PerfTracker
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)

_perf = PerfTracker.get("shm_connector")


class SharedMemoryConnector(OmniConnectorBase):
    """
    Connector that uses SharedMemory for large objects and inline data for small objects.
    Acts as a unified replacement for the legacy IPC fallback logic.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.stage_id = config.get("stage_id", -1)
        self.device = config.get("device", "cuda:0")
        self.threshold = int(config.get("shm_threshold_bytes", 65536))
        self._metrics = {
            "puts": 0,
            "gets": 0,
            "bytes_transferred": 0,
            "shm_writes": 0,
            "inline_writes": 0,
        }

    def put(
        self,
        from_stage: str,
        to_stage: str,
        put_key: str,
        data: Any,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        t_put_start = time.monotonic()
        try:
            t0 = time.monotonic()
            payload = self.serialize_obj(data)
            _perf.record("put_serialize", (time.monotonic() - t0) * 1000)
            size = len(payload)
            _perf.record("put_payload_bytes", size)

            if True:
                lock_file = f"/dev/shm/shm_{put_key}_lockfile.lock"
                t_lock_start = time.monotonic()
                with open(lock_file, "wb+") as lockf:
                    fcntl.flock(lockf, fcntl.LOCK_EX)
                    _perf.record("put_lock_wait", (time.monotonic() - t_lock_start) * 1000)
                    t_write = time.monotonic()
                    meta = shm_write_bytes(payload, name=put_key)
                    _perf.record("put_shm_write", (time.monotonic() - t_write) * 1000)
                    fcntl.flock(lockf, fcntl.LOCK_UN)

                metadata = {"shm": meta, "size": size}
                self._metrics["shm_writes"] += 1
            else:
                metadata = {"inline_bytes": payload, "size": size}
                self._metrics["inline_writes"] += 1

            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += size

            _perf.record("put_total", (time.monotonic() - t_put_start) * 1000)
            return True, size, metadata

        except Exception as e:
            logger.error(f"SharedMemoryConnector put failed for req {put_key}: {e}")
            return False, 0, None

    def _get_data_with_lock(self, lock_file: str, shm_handle: dict):
        obj = None
        try:
            t_lock_start = time.monotonic()
            with open(lock_file, "rb+") as lockf:
                fcntl.flock(lockf, fcntl.LOCK_EX)
                _perf.record("get_lock_wait", (time.monotonic() - t_lock_start) * 1000)
                t_read = time.monotonic()
                data_bytes = shm_read_bytes(shm_handle)
                _perf.record("get_shm_read", (time.monotonic() - t_read) * 1000)
                fcntl.flock(lockf, fcntl.LOCK_UN)
            t_deser = time.monotonic()
            obj = self.deserialize_obj(data_bytes)
            _perf.record("get_deserialize", (time.monotonic() - t_deser) * 1000)
            _perf.record("get_payload_bytes", int(shm_handle.get("size", 0)))
            return obj, int(shm_handle.get("size", 0))
        except Exception as e:
            logger.error(f"SharedMemoryConnector shm get failed for req : {e}")
            return None
        finally:
            if obj and os.path.exists(lock_file):
                os.remove(lock_file)

    def get(
        self,
        from_stage: str,
        to_stage: str,
        get_key: str,
        metadata=None,
    ) -> tuple[Any, int] | None:
        t_get_start = time.monotonic()
        if metadata is not None:
            if isinstance(metadata, dict) and get_key in metadata:
                metadata = metadata.get(get_key)

            if not isinstance(metadata, dict):
                return None

            if "inline_bytes" in metadata:
                try:
                    t0 = time.monotonic()
                    obj = self.deserialize_obj(metadata["inline_bytes"])
                    _perf.record("get_inline_deserialize", (time.monotonic() - t0) * 1000)
                    _perf.record("get_total", (time.monotonic() - t_get_start) * 1000)
                    return obj, int(metadata.get("size", 0))
                except Exception as e:
                    logger.error(f"SharedMemoryConnector inline get failed for req {get_key}: {e}")
                    return None

            if "shm" in metadata:
                shm_handle = metadata["shm"]
                lock_file = f"/dev/shm/shm_{shm_handle['name']}_lockfile.lock"
                result = self._get_data_with_lock(lock_file, shm_handle)
                _perf.record("get_total", (time.monotonic() - t_get_start) * 1000)
                return result

            return None
        shm = None
        try:
            shm = shm_pkg.SharedMemory(name=get_key)
            if shm is None or shm.size == 0:
                return None
            lock_file = f"/dev/shm/shm_{get_key}_lockfile.lock"
            shm_handle = {"name": get_key, "size": shm.size}
            result = self._get_data_with_lock(lock_file, shm_handle)
            _perf.record("get_total", (time.monotonic() - t_get_start) * 1000)
            return result
        except Exception:
            return None
        finally:
            if shm:
                shm.close()

    def cleanup(self, request_id: str) -> None:
        # SHM segments are automatically unlinked during 'get' (shm_read_bytes).
        # If 'get' is never called (e.g. error flow), the SHM segment might leak.
        # A robust implementation might track created segments and unlink them here
        # if they haven't been consumed.
        # For now, we rely on the consumer to read and unlink.
        pass

    def close(self) -> None:
        pass

    def health(self) -> dict[str, Any]:
        return {"status": "healthy", "threshold": self.threshold, **self._metrics}

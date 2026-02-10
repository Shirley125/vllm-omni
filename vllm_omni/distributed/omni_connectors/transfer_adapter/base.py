# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time
from typing import Any

from ..utils.logging import get_connector_logger

logger = get_connector_logger(__name__)


class OmniTransferAdapterBase:
    """Base class for managing data transfer via OmniConnector.

    This class handles the core loop logic and connector interactions, but
    leaves the specific data processing (chunks, KV cache, etc.) to subclasses.
    """

    def __init__(self, config: Any):
        self.config = config
        if not hasattr(self, "connector"):
            self.connector = None
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
        self._shutdown_lock = threading.Lock()
        self._is_shutdown = False
        self._connector_closed = False

        self.recv_thread = threading.Thread(
            target=self.recv_loop,
            daemon=True,
            name=f"{self.__class__.__name__}-recv",
        )
        self.recv_thread.start()

        self.save_thread = threading.Thread(
            target=self.save_loop,
            daemon=True,
            name=f"{self.__class__.__name__}-save",
        )
        self.save_thread.start()

    @classmethod
    def create_connector(cls, model_config: Any):
        raise NotImplementedError

    def recv_loop(self):
        """Loop to poll for incoming data."""
        while not self.stop_event.is_set():
            # Iterate over a snapshot of pending requests
            with self.lock:
                pending_reqs_ids = list(self._pending_load_reqs.keys())

            for req_id in pending_reqs_ids:
                try:
                    self._poll_single_request(req_id)
                except Exception as e:
                    logger.warning(f"Error receiving data for {req_id}: {e}")

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
                    self._send_single_request(task)
                except Exception as e:
                    logger.error(f"Error saving data for {task.get('request_id')}: {e}")
            else:
                time.sleep(0.001)

    def _poll_single_request(self, *args, **kwargs):
        """Poll connector for a single request task.
        Subclasses should implement request-specific receive behavior."""
        raise NotImplementedError

    def _send_single_request(self, *args, **kwargs):
        """Send one pending save request task to the connector.
        Subclasses should implement task-specific handling logic."""
        raise NotImplementedError

    def load_async(self, *args, **kwargs):
        """Register a request to load data. To be implemented by subclasses."""
        raise NotImplementedError

    def save_async(self, *args, **kwargs):
        """Submit data to be saved. To be implemented by subclasses."""
        raise NotImplementedError

    def load(self, *args, **kwargs):
        """Load request data from connector synchronously. To be implemented by subclasses."""
        raise NotImplementedError

    def save(self, *args, **kwargs):
        """Save data to connector synchronously. To be implemented by subclasses."""
        raise NotImplementedError

    def get_finished_requests(self):
        """Get finished loaded or saved requests"""
        raise NotImplementedError

    def shutdown(self, close_connector: bool = True, join_timeout: float | None = 5.0) -> None:
        """Stop background loops and optionally close connector resources.

        This method is idempotent and safe to call multiple times.
        """
        should_join_threads = False
        with self._shutdown_lock:
            if not self._is_shutdown:
                self._is_shutdown = True
                self.stop_event.set()
                should_join_threads = True

        if should_join_threads:
            current_ident = threading.get_ident()
            for thread_name in ("recv_thread", "save_thread"):
                thread = getattr(self, thread_name, None)
                if thread is None:
                    continue
                # Avoid deadlocking if shutdown is called from inside the worker thread.
                if thread.ident == current_ident:
                    continue
                thread.join(timeout=join_timeout)
                if thread.is_alive():
                    logger.warning(
                        "Timed out joining %s for %s",
                        thread_name,
                        self.__class__.__name__,
                    )

            with self.lock:
                self._pending_load_reqs.clear()
                self._finished_load_reqs.clear()
                self._pending_save_reqs.clear()
                self._finished_save_reqs.clear()

        if close_connector:
            with self._shutdown_lock:
                if self._connector_closed:
                    return
                self._connector_closed = True
            connector = getattr(self, "connector", None)
            close_func = getattr(connector, "close", None) if connector is not None else None
            if callable(close_func):
                try:
                    close_func()
                except Exception:
                    logger.exception("Failed closing connector for %s", self.__class__.__name__)

    def close(self, close_connector: bool = True, join_timeout: float | None = 5.0) -> None:
        """Alias of :meth:`shutdown` for compatibility."""
        self.shutdown(close_connector=close_connector, join_timeout=join_timeout)

    def __del__(self) -> None:
        # Best-effort cleanup; avoid raising in GC paths.
        try:
            self.shutdown()
        except Exception:
            pass

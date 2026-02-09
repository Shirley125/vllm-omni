# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import threading
import time
from typing import Any

from ..utils.logging import get_connector_logger

logger = get_connector_logger(__name__)


class OmniModelMode(enum.Enum):
    # Omni AR Model
    MODE_AR = "ar"

    # Omni Generation Model
    MODE_GENERATION = "generate"


class OmniTransferManagerBase:
    """Base class for managing asynchronous data transfer via OmniConnector.

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

        self.recv_thread = threading.Thread(target=self.recv_loop, daemon=True)
        self.recv_thread.start()

        self.save_thread = threading.Thread(target=self.save_loop, daemon=True)
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
                    self._send_single_task(task)
                except Exception as e:
                    logger.error(f"Error sending data for {task.get('request_id')}: {e}")
            else:
                time.sleep(0.001)

    def _poll_single_request(self, req_id: str):
        """Poll connector for a single request ID (non-blocking).

        Subclasses should implement request-specific receive/merge behavior.
        """
        raise NotImplementedError

    def _send_single_task(self, task: dict):
        """Send one pending task to the connector.

        Subclasses should implement task-specific serialization and routing.
        """
        raise NotImplementedError

    def load(self, *args, **kwargs):
        """Register a request to load data. To be implemented by subclasses."""
        raise NotImplementedError

    def save(self, *args, **kwargs):
        """Submit data to be saved/sent. To be implemented by subclasses."""
        raise NotImplementedError

    def get_finished_requests(self):
        raise NotImplementedError

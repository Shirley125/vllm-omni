# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base class for chunk and cache transfer management."""

import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .connectors.base import OmniConnectorBase

from .utils.logging import get_connector_logger

logger = get_connector_logger(__name__)


class BaseOmniChunkManager(ABC):
    """Base class for managing asynchronous chunk/cache operations via OmniConnector.
    
    This class provides common infrastructure for:
    - Connection management
    - Request queue management (pending/finished)
    - Background thread management for async I/O
    - Basic lifecycle management
    
    Subclasses should implement specific logic for:
    - Processing loaded chunks
    - Preparing chunks for saving
    - Custom request handling
    """

    def __init__(self, connector: "OmniConnectorBase"):
        """Initialize base chunk manager.
        
        Args:
            connector: The OmniConnector instance for data transfer
        """
        self.connector = connector
        
        # Load request tracking
        self._pending_load_reqs: dict[str, Any] = {}
        self._finished_load_reqs: set[str] = set()
        
        # Save request tracking
        self._pending_save_reqs: dict[str, Any] = {}
        self._finished_save_reqs: set[str] = set()
        
        # Thread management
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        
        # Start background threads
        self.recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.recv_thread.start()
        
        self.save_thread = threading.Thread(target=self._save_loop, daemon=True)
        self.save_thread.start()
        
        logger.info(f"{self.__class__.__name__} initialized with connector: {connector.__class__.__name__}")

    def stop(self):
        """Stop background threads gracefully."""
        logger.info(f"Stopping {self.__class__.__name__}")
        self.stop_event.set()
        if self.recv_thread.is_alive():
            self.recv_thread.join(timeout=2.0)
        if self.save_thread.is_alive():
            self.save_thread.join(timeout=2.0)

    # ========================================================================
    # Public Interface - Scheduler Integration
    # ========================================================================
    
    def get_finished_load_requests(self) -> set[str]:
        """Get and clear requests that have finished loading.
        
        Returns:
            Set of request IDs that have successfully loaded chunks
        """
        with self.lock:
            finished = set(self._finished_load_reqs)
            self._finished_load_reqs.clear()
        return finished

    def get_finished_save_requests(self) -> set[str]:
        """Get and clear requests that have finished saving.
        
        Returns:
            Set of request IDs that have successfully saved chunks
        """
        with self.lock:
            finished = set(self._finished_save_reqs)
            self._finished_save_reqs.clear()
        return finished

    @abstractmethod
    def request_chunk(self, request: Any) -> None:
        """Request a chunk to be loaded asynchronously.
        
        Subclasses should implement this to:
        1. Validate the request
        2. Add to pending_load_reqs
        3. Initialize any request-specific state
        
        Args:
            request: The request object needing a chunk
        """
        pass

    @abstractmethod
    def submit_chunk(self, output: Any, request: Any, custom_process_func=None) -> None:
        """Submit a chunk to be saved asynchronously.
        
        Subclasses should implement this to:
        1. Process the output data
        2. Add to pending_save_reqs
        3. Handle any request-specific logic
        
        Args:
            output: The output data to save
            request: The request object
            custom_process_func: Optional custom processing function
        """
        pass

    # ========================================================================
    # Internal Async I/O Methods
    # ========================================================================

    def _recv_loop(self):
        """Background thread loop for receiving chunks.
        
        Continuously polls for pending load requests and attempts to
        receive data from the connector.
        """
        while not self.stop_event.is_set():
            try:
                # Get snapshot of pending requests
                with self.lock:
                    pending_req_ids = list(self._pending_load_reqs.keys())

                # Process each pending request
                for req_id in pending_req_ids:
                    try:
                        self._process_load_request(req_id)
                    except Exception as e:
                        logger.warning(f"Error processing load request {req_id}: {e}")

            except Exception as e:
                logger.error(f"Error in recv_loop: {e}")

            # Small sleep to avoid busy-waiting
            self.stop_event.wait(timeout=0.001)

    def _save_loop(self):
        """Background thread loop for saving chunks.
        
        Continuously processes pending save requests from the queue.
        """
        while not self.stop_event.is_set():
            try:
                task = self._get_next_save_task()
                if task:
                    self._process_save_task(task)
                else:
                    # No tasks, sleep briefly
                    self.stop_event.wait(timeout=0.001)
            except Exception as e:
                logger.error(f"Error in save_loop: {e}")

    @abstractmethod
    def _process_load_request(self, req_id: str) -> None:
        """Process a single load request.
        
        Subclasses should implement:
        1. Build the connector key
        2. Call connector.get()
        3. Process received data
        4. Update request state
        5. Move from pending to finished
        
        Args:
            req_id: Request ID to process
        """
        pass

    @abstractmethod
    def _get_next_save_task(self) -> dict[str, Any] | None:
        """Get the next save task from the queue.
        
        Subclasses should implement logic to:
        1. Get next task from pending_save_reqs
        2. Return task dict or None if queue empty
        
        Returns:
            Task dictionary or None
        """
        pass

    @abstractmethod
    def _process_save_task(self, task: dict[str, Any]) -> None:
        """Process a single save task.
        
        Subclasses should implement:
        1. Extract task data
        2. Call connector.put()
        3. Update state on success/failure
        
        Args:
            task: Task dictionary with save parameters
        """
        pass

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _mark_load_finished(self, req_id: str) -> None:
        """Mark a load request as finished.
        
        Args:
            req_id: Request ID to mark as finished
        """
        with self.lock:
            self._finished_load_reqs.add(req_id)
            if req_id in self._pending_load_reqs:
                del self._pending_load_reqs[req_id]

    def _mark_save_finished(self, req_id: str) -> None:
        """Mark a save request as finished.
        
        Args:
            req_id: Request ID to mark as finished
        """
        with self.lock:
            self._finished_save_reqs.add(req_id)

    def __del__(self):
        """Cleanup on destruction."""
        if not self.stop_event.is_set():
            self.stop()

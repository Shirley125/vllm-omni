# SPDX-License-Identifier: Apache-2.0

from collections import deque
from typing import Any

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request, RequestStatus

from vllm_omni.distributed.omni_connectors.adapter import OmniChunkManager


class ChunkRequestProcessor:
    """
    Processor for handling chunk-based requests.
    Decouples chunk processing logic from the main scheduler.
    """

    def __init__(self, chunk_manager: OmniChunkManager):
        self.chunk_manager = chunk_manager
        self.waiting_for_chunk_waiting_requests: deque[Request] = deque()
        self.waiting_for_chunk_running_requests: deque[Request] = deque()
        self.finished_load_chunk_reqs = set()
        self.requests_with_ready_chunks = set()

    def process_pending_chunks(
        self,
        waiting_queue: Any,
        running_queue: list[Request],
    ) -> int:
        """
        Process pending chunks for waiting and running queues.
        Returns the number of running requests waiting for chunks.
        """
        self.finished_load_chunk_reqs = self.chunk_manager.get_finished()
        self._process_chunk_queue(waiting_queue, self.waiting_for_chunk_waiting_requests, RequestStatus.WAITING)
        self._process_chunk_queue(
            running_queue,
            self.waiting_for_chunk_running_requests,
            RequestStatus.RUNNING,
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

        self.finished_load_chunk_reqs = set()

    def filter_scheduler_output(self, scheduler_output: SchedulerOutput) -> None:
        """
        Clean up ready chunks from scheduler output.
        """
        self._clear_chunk_ready(scheduler_output)

    def _process_chunk_queue(
        self,
        queue: Any,
        waiting_for_chunk_list: deque[Request],
        target_status: RequestStatus,
    ) -> None:
        queue_snapshot = list(queue)
        for request in queue_snapshot:
            if request.status != RequestStatus.WAITING_FOR_CHUNK:
                if request.request_id in self.requests_with_ready_chunks:
                    continue
                # Access finished_requests from chunk_manager instead of connector
                if request.request_id in self.chunk_manager.finished_requests:
                    self.chunk_manager.finished_requests.remove(request.request_id)
                    request.additional_information = None
                    continue
                self.chunk_manager.get_chunk(request)
                request.status = RequestStatus.WAITING_FOR_CHUNK
            else:
                if request.request_id in self.finished_load_chunk_reqs:
                    request.status = target_status
                    self.requests_with_ready_chunks.add(request.request_id)
                    continue
            queue.remove(request)
            waiting_for_chunk_list.append(request)

    def _clear_chunk_ready(self, scheduler_output: SchedulerOutput) -> None:
        if scheduler_output.scheduled_new_reqs:
            for req_data in scheduler_output.scheduled_new_reqs:
                if req_data.req_id in self.requests_with_ready_chunks:
                    self.requests_with_ready_chunks.remove(req_data.req_id)

        if scheduler_output.scheduled_cached_reqs:
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                if req_id in self.requests_with_ready_chunks:
                    self.requests_with_ready_chunks.remove(req_id)

from __future__ import annotations


class OmniSchedulerMixin:
    """Shared scheduler helpers for omni-specific request handling."""

    def _free_input_coordinator_request(self, request_id: str) -> None:
        """Prune full-payload coordinator state for a completed request."""
        input_coordinator = getattr(self, "input_coordinator", None)
        if input_coordinator is not None:
            input_coordinator.free_finished_request(request_id)

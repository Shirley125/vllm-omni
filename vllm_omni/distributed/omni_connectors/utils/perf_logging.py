# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Lightweight performance tracker for async-chunk pipeline profiling.

Usage:
    from vllm_omni.distributed.omni_connectors.utils.perf_logging import PerfTracker

    tracker = PerfTracker.get("shm_connector")
    tracker.record("put_serialize", elapsed_ms)
    # Stats are printed periodically (every ``report_interval`` records).
"""

import threading
import time
from collections import defaultdict
from typing import ClassVar

from .logging import get_connector_logger

logger = get_connector_logger("perf_tracker")

_REPORT_INTERVAL_SECONDS = 10.0


class _MetricBucket:
    """Thread-safe accumulator for a single named metric."""

    __slots__ = ("count", "total", "min_val", "max_val", "lock")

    def __init__(self):
        self.count: int = 0
        self.total: float = 0.0
        self.min_val: float = float("inf")
        self.max_val: float = 0.0
        self.lock = threading.Lock()

    def record(self, value: float) -> None:
        with self.lock:
            self.count += 1
            self.total += value
            if value < self.min_val:
                self.min_val = value
            if value > self.max_val:
                self.max_val = value

    def snapshot_and_reset(self) -> dict | None:
        with self.lock:
            if self.count == 0:
                return None
            result = {
                "count": self.count,
                "total_ms": round(self.total, 3),
                "avg_ms": round(self.total / self.count, 3),
                "min_ms": round(self.min_val, 3),
                "max_ms": round(self.max_val, 3),
            }
            self.count = 0
            self.total = 0.0
            self.min_val = float("inf")
            self.max_val = 0.0
            return result


class PerfTracker:
    """Per-component performance tracker with periodic console reporting."""

    _instances: ClassVar[dict[str, "PerfTracker"]] = {}
    _instances_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, name: str, report_interval: float = _REPORT_INTERVAL_SECONDS):
        self.name = name
        self.report_interval = report_interval
        self._buckets: dict[str, _MetricBucket] = defaultdict(_MetricBucket)
        self._last_report = time.monotonic()
        self._report_lock = threading.Lock()

    @classmethod
    def get(cls, name: str, report_interval: float = _REPORT_INTERVAL_SECONDS) -> "PerfTracker":
        with cls._instances_lock:
            if name not in cls._instances:
                cls._instances[name] = cls(name, report_interval)
            return cls._instances[name]

    def record(self, metric: str, elapsed_ms: float) -> None:
        self._buckets[metric].record(elapsed_ms)
        self._maybe_report()

    def _maybe_report(self) -> None:
        now = time.monotonic()
        if now - self._last_report < self.report_interval:
            return
        if not self._report_lock.acquire(blocking=False):
            return
        try:
            self._last_report = now
            self._do_report()
        finally:
            self._report_lock.release()

    def _do_report(self) -> None:
        lines = [f"[PerfTracker:{self.name}] === Performance Report ==="]
        any_data = False
        for metric_name in sorted(self._buckets.keys()):
            snap = self._buckets[metric_name].snapshot_and_reset()
            if snap is None:
                continue
            any_data = True
            lines.append(
                f"  {metric_name}: "
                f"count={snap['count']}, "
                f"avg={snap['avg_ms']:.3f}ms, "
                f"min={snap['min_ms']:.3f}ms, "
                f"max={snap['max_ms']:.3f}ms, "
                f"total={snap['total_ms']:.3f}ms"
            )
        if any_data:
            logger.warning("\n".join(lines))

    def force_report(self) -> None:
        with self._report_lock:
            self._do_report()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Lightweight performance tracker for async-chunk pipeline profiling.

Usage:
    from vllm_omni.distributed.omni_connectors.utils.perf_logging import PerfTracker

    tracker = PerfTracker.get("shm_connector")
    tracker.record("put_serialize", elapsed_ms)

Two reporting modes:
- **Periodic (window)**: every ``report_interval`` seconds, prints the stats
  collected in that window then resets the window counters.  Useful for live
  monitoring while the benchmark is running.
- **Cumulative (final)**: accumulates across the entire process lifetime.
  Automatically printed via ``atexit`` when the process exits, giving the
  full-run average.
"""

import atexit
import threading
import time
from collections import defaultdict
from typing import ClassVar

from .logging import get_connector_logger

logger = get_connector_logger("perf_tracker")

_REPORT_INTERVAL_SECONDS = 30.0


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
            result = self._format_unlocked()
            self.count = 0
            self.total = 0.0
            self.min_val = float("inf")
            self.max_val = 0.0
            return result

    def snapshot(self) -> dict | None:
        with self.lock:
            if self.count == 0:
                return None
            return self._format_unlocked()

    def _format_unlocked(self) -> dict:
        return {
            "count": self.count,
            "total_ms": round(self.total, 3),
            "avg_ms": round(self.total / self.count, 3),
            "min_ms": round(self.min_val, 3),
            "max_ms": round(self.max_val, 3),
        }


class PerfTracker:
    """Per-component performance tracker with periodic + cumulative reporting."""

    _instances: ClassVar[dict[str, "PerfTracker"]] = {}
    _instances_lock: ClassVar[threading.Lock] = threading.Lock()
    _atexit_registered: ClassVar[bool] = False

    def __init__(self, name: str, report_interval: float = _REPORT_INTERVAL_SECONDS):
        self.name = name
        self.report_interval = report_interval
        # Window buckets: reset after each periodic report
        self._window: dict[str, _MetricBucket] = defaultdict(_MetricBucket)
        # Cumulative buckets: never reset, for final report
        self._cumulative: dict[str, _MetricBucket] = defaultdict(_MetricBucket)
        self._last_report = time.monotonic()
        self._report_lock = threading.Lock()

    @classmethod
    def get(cls, name: str, report_interval: float = _REPORT_INTERVAL_SECONDS) -> "PerfTracker":
        with cls._instances_lock:
            if not cls._atexit_registered:
                atexit.register(cls._atexit_report_all)
                cls._atexit_registered = True
            if name not in cls._instances:
                cls._instances[name] = cls(name, report_interval)
            return cls._instances[name]

    def record(self, metric: str, value: float) -> None:
        self._window[metric].record(value)
        self._cumulative[metric].record(value)
        self._maybe_report()

    # ------------------------------------------------------------------
    # Periodic window report
    # ------------------------------------------------------------------

    def _maybe_report(self) -> None:
        now = time.monotonic()
        if now - self._last_report < self.report_interval:
            return
        if not self._report_lock.acquire(blocking=False):
            return
        try:
            self._last_report = now
            self._do_window_report()
        finally:
            self._report_lock.release()

    def _do_window_report(self) -> None:
        lines = [f"[PerfTracker:{self.name}] --- Window Report (last {self.report_interval}s) ---"]
        any_data = False
        for metric_name in sorted(self._window.keys()):
            snap = self._window[metric_name].snapshot_and_reset()
            if snap is None:
                continue
            any_data = True
            lines.append(self._format_line(metric_name, snap))
        if any_data:
            logger.warning("\n".join(lines))

    # ------------------------------------------------------------------
    # Cumulative (full-run) report
    # ------------------------------------------------------------------

    def cumulative_report(self) -> None:
        lines = [f"[PerfTracker:{self.name}] === CUMULATIVE (full-run) Report ==="]
        any_data = False
        for metric_name in sorted(self._cumulative.keys()):
            snap = self._cumulative[metric_name].snapshot()
            if snap is None:
                continue
            any_data = True
            lines.append(self._format_line(metric_name, snap))
        if any_data:
            logger.warning("\n".join(lines))

    @classmethod
    def _atexit_report_all(cls) -> None:
        with cls._instances_lock:
            instances = list(cls._instances.values())
        if not instances:
            return
        logger.warning("=" * 72)
        logger.warning("[PerfTracker] FINAL CUMULATIVE REPORTS (full run)")
        logger.warning("=" * 72)
        for tracker in instances:
            tracker.cumulative_report()
        logger.warning("=" * 72)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_line(metric_name: str, snap: dict) -> str:
        return (
            f"  {metric_name}: "
            f"count={snap['count']}, "
            f"avg={snap['avg_ms']:.3f}ms, "
            f"min={snap['min_ms']:.3f}ms, "
            f"max={snap['max_ms']:.3f}ms, "
            f"total={snap['total_ms']:.3f}ms"
        )

    def force_report(self) -> None:
        """Print both window and cumulative reports immediately."""
        with self._report_lock:
            self._do_window_report()
        self.cumulative_report()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Lightweight per-call performance logger for async-chunk pipeline profiling.

Every ``record()`` call emits one log line to stderr in a fixed format that
can be parsed by the companion ``parse_perf_log.py`` script.

Format:
    [PERF] <component> | <metric> | <value>

Usage:
    from vllm_omni.distributed.omni_connectors.utils.perf_logging import PerfTracker

    tracker = PerfTracker.get("shm_connector")
    tracker.record("put_serialize", elapsed_ms)
"""

import sys
import time


class PerfTracker:
    _instances: dict[str, "PerfTracker"] = {}

    def __init__(self, name: str):
        self.name = name

    @classmethod
    def get(cls, name: str) -> "PerfTracker":
        if name not in cls._instances:
            cls._instances[name] = cls(name)
        return cls._instances[name]

    def record(self, metric: str, value: float) -> None:
        # Fast path: write directly to stderr, bypass logging framework
        sys.stderr.write(f"[PERF] {self.name} | {metric} | {value:.4f}\n")

    @staticmethod
    def now() -> float:
        """Return current monotonic time in seconds (convenience helper)."""
        return time.monotonic()

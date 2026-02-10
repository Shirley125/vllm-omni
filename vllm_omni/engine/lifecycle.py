# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

_PRIMITIVE_TYPES = (str, bytes, int, float, bool, type(None))
_KNOWN_CHILD_ATTRS = (
    "llm_engine",
    "engine_core",
    "scheduler",
    "schedulers",
    "_scheduler",
    "_schedulers",
    "executor",
    "model_executor",
)


def shutdown_chunk_transfer_adapters(
    owner: Any,
    logger: Any,
    *,
    max_depth: int = 6,
    max_nodes: int = 512,
) -> int:
    """Best-effort shutdown for reachable chunk transfer adapters."""
    if owner is None:
        return 0

    stack: list[tuple[Any, int]] = [(owner, 0)]
    visited: set[int] = set()
    shutdown_adapters: set[int] = set()
    closed_count = 0

    while stack and len(visited) < max_nodes:
        current, depth = stack.pop()
        if isinstance(current, _PRIMITIVE_TYPES):
            continue

        current_id = id(current)
        if current_id in visited:
            continue
        visited.add(current_id)

        adapter = None
        current_dict = getattr(current, "__dict__", None)
        if isinstance(current_dict, dict):
            adapter = current_dict.get("chunk_transfer_adapter")
        if adapter is None:
            try:
                adapter = getattr(current, "chunk_transfer_adapter", None)
            except Exception:
                adapter = None

        if adapter is not None:
            adapter_id = id(adapter)
            if adapter_id not in shutdown_adapters:
                shutdown_adapters.add(adapter_id)
                shutdown_fn = getattr(adapter, "shutdown", None)
                if callable(shutdown_fn):
                    try:
                        shutdown_fn()
                        closed_count += 1
                    except Exception:
                        logger.exception("Failed to shutdown chunk transfer adapter.")
            try:
                setattr(current, "chunk_transfer_adapter", None)
            except Exception:
                pass

        if depth >= max_depth:
            continue

        children: list[Any] = []
        if isinstance(current, dict):
            children.extend(current.values())
        elif isinstance(current, (list, tuple, set, frozenset)):
            children.extend(current)
        elif isinstance(current_dict, dict):
            children.extend(current_dict.values())

        for attr_name in _KNOWN_CHILD_ATTRS:
            try:
                child = getattr(current, attr_name, None)
            except Exception:
                continue
            if child is not None:
                children.append(child)

        for child in children:
            if isinstance(child, _PRIMITIVE_TYPES):
                continue
            stack.append((child, depth + 1))

    return closed_count

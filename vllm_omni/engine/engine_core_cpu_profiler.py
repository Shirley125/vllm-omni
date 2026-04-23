# SPDX-License-Identifier: Apache-2.0
"""
CPU Kineto trace in the *EngineCore* process (scheduler) when ``profiler_config.profiler
== "torch"``.

Why this module exists
----------------------
``/start_profile`` and ``AsyncOmni.start_profile`` use ``call_utility("collective_rpc", ...)``,
which the engine handles by calling :meth:`vllm.v1.engine.core.EngineCore.collective_rpc`.
That call **never** reached overrides on :class:`StageEngineCoreProc` when the subprocess was
vendored :class:`vllm.v1.engine.core.EngineCoreProc` (e.g. ``CoreEngineProcManager`` /
single-stage launch), because only :class:`EngineCore` defines ``collective_rpc``.

We therefore **patch** :class:`EngineCore` once at import time so *every* engine subprocess
(``StageEngineCoreProc`` or upstream ``EngineCoreProc``) gets the same hook.

``collective_rpc_async`` lives only on the *client* (``AsyncMPClient``); the child only runs
synchronous ``EngineCore.collective_rpc``.
"""

from __future__ import annotations

import functools
import os
import time
from collections.abc import Callable
from typing import Any

import torch
from vllm.config.profiler import _is_uri_path
from vllm.logger import init_logger

logger = init_logger(__name__)

_ATTR = "_omni_engine_cpu_torch_profiler"


def _resolve_engine_profiler_config(engine_core: Any) -> Any:
    vllm = getattr(engine_core, "vllm_config", None)
    if vllm is not None:
        pc = getattr(vllm, "profiler_config", None)
        if pc is not None:
            return pc
    ex = getattr(engine_core, "model_executor", None)
    if ex is not None:
        v2 = getattr(ex, "vllm_config", None)
        if v2 is not None:
            return getattr(v2, "profiler_config", None)
    return None


def _engine_profile_flags(engine_core: Any) -> tuple[Any, Any, str, bool, bool]:
    prof_top = getattr(getattr(engine_core, "vllm_config", None), "profiler_config", None)
    prof_cfg = _resolve_engine_profiler_config(engine_core)
    if prof_cfg is not None:
        tdir = getattr(prof_cfg, "torch_profiler_dir", None) or ""
    else:
        tdir = ""
    tdir_is_uri = bool(tdir) and _is_uri_path(tdir)
    use_engine_cpu = (
        prof_cfg is not None
        and getattr(prof_cfg, "profiler", None) == "torch"
        and bool(tdir)
        and not tdir_is_uri
    )
    return prof_top, prof_cfg, tdir, tdir_is_uri, use_engine_cpu


def _log_decision(
    where: str, engine_core: Any, is_start: bool, prof_top: Any, prof_cfg: Any, tdir: str, tdir_is_uri: bool, use: bool
) -> None:
    mc = getattr(getattr(engine_core, "vllm_config", None), "model_config", None)
    stage_id = getattr(mc, "stage_id", None) if mc is not None else None
    logger.info(
        "[engine_core profile] from=%s is_start=%s stage_id=%s vllm_config.profiler_config_set=%s "
        "resolved_profiler_config_set=%s profiler=%r torch_profiler_dir_set=%s use_engine_cpu=%s",
        where,
        is_start,
        stage_id,
        prof_top is not None,
        prof_cfg is not None,
        getattr(prof_cfg, "profiler", None) if prof_cfg is not None else None,
        bool(tdir) and not tdir_is_uri,
        use,
    )
    if where == "EngineCore.collective_rpc":
        # Extra visibility: profile path is easy to miss in log aggregation
        logger.warning(
            "[OMNI] engine CPU profile hook: %s (use_engine_cpu=%s stage_id=%s)",
            "start" if is_start else "stop",
            use,
            stage_id,
        )


def _start_engine_cpu_profiler(self: Any, prof_cfg: Any, profile_prefix: str | None) -> None:
    if getattr(self, _ATTR, None) is not None:
        logger.warning("Engine core torch profiler already running; skipping duplicate start")
        return
    out_dir = getattr(prof_cfg, "torch_profiler_dir", None) or ""
    if not out_dir or _is_uri_path(out_dir):
        return
    os.makedirs(out_dir, exist_ok=True)
    mc = getattr(self.vllm_config, "model_config", None)
    stage_id = getattr(mc, "stage_id", 0) if mc is not None else 0
    ts = int(time.time())
    stem = profile_prefix or f"engine_core_stage{stage_id}_{ts}"
    if os.path.dirname(stem):
        session_dir = os.path.dirname(stem) or out_dir
        file_stem = os.path.basename(stem)
    else:
        session_dir = os.path.join(out_dir, f"{ts}_engine_core_stage{stage_id}")
        file_stem = stem
    os.makedirs(session_dir, exist_ok=True)
    trace_path = os.path.join(session_dir, f"{file_stem}.json")

    def on_trace_ready(prof: Any) -> None:
        try:
            prof.export_chrome_trace(trace_path)
            logger.info("Engine/scheduler trace (CPU) saved to: %s", trace_path)
        except Exception as err:
            logger.warning("Failed to export engine/scheduler trace: %s", err)

    prof_obj = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=bool(getattr(prof_cfg, "torch_profiler_record_shapes", False)),
        profile_memory=bool(getattr(prof_cfg, "torch_profiler_with_memory", False)),
        with_stack=bool(getattr(prof_cfg, "torch_profiler_with_stack", False)),
        with_flops=bool(getattr(prof_cfg, "torch_profiler_with_flops", False)),
        on_trace_ready=on_trace_ready,
    )
    prof_obj.start()
    setattr(self, _ATTR, prof_obj)
    logger.info("Engine core (scheduler) torch profiler started; trace -> %s", trace_path)


def _stop_engine_cpu_profiler(self: Any) -> None:
    prof = getattr(self, _ATTR, None)
    if prof is None:
        return
    try:
        prof.stop()
    except Exception as err:
        logger.warning("Error stopping engine core torch profiler: %s", err)
    finally:
        setattr(self, _ATTR, None)


def _run_with_engine_cpu_profile(
    self: Any,
    is_start: bool,
    profile_prefix: str | None,
    from_where: str,
    run_workers: Callable[[], Any],
) -> Any:
    prof_top, prof_cfg, tdir, tdir_is_uri, use_engine_cpu = _engine_profile_flags(self)
    _log_decision(from_where, self, is_start, prof_top, prof_cfg, tdir, tdir_is_uri, use_engine_cpu)
    if is_start:
        if use_engine_cpu and prof_cfg is not None:
            _start_engine_cpu_profiler(self, prof_cfg, profile_prefix)
        try:
            return run_workers()
        except Exception:
            if use_engine_cpu or getattr(self, _ATTR, None) is not None:
                _stop_engine_cpu_profiler(self)
            raise
    out = run_workers()
    if use_engine_cpu or getattr(self, _ATTR, None) is not None:
        _stop_engine_cpu_profiler(self)
    return out


def _inner_args_ok(args: object) -> bool:
    if args is None:
        return False
    if isinstance(args, (list, tuple)):
        return len(args) > 0
    return False


def wrap_engine_core_collective_rpc(
    orig: Callable[..., list[Any]],
    self: object,
    method: str | Any,
    timeout: float | None = None,
    args: tuple = (),
    kwargs: dict[str, Any] | None = None,
) -> list[Any]:
    method_s = method if isinstance(method, str) else str(method)
    inner = () if args is None else args
    n = len(inner) if isinstance(inner, (list, tuple)) else 0
    is_profile = method_s == "profile" and n > 0
    if is_profile and isinstance(inner, (list, tuple)):
        is_start = bool(inner[0])
        prof_prefix: str | None = inner[1] if n > 1 else None

        def _run() -> list[Any]:
            return orig(
                self,
                method,
                timeout,
                args,
                kwargs or {},
            )

        return _run_with_engine_cpu_profile(
            self,
            is_start,
            prof_prefix,
            "EngineCore.collective_rpc",
            _run,
        )
    if method_s == "profile" and not _inner_args_ok(args):
        logger.warning(
            "[OMNI] EngineCore.collective_rpc('profile') with empty/invalid args=%r; "
            "not wrapping engine CPU profiler",
            args,
        )
    return orig(self, method, timeout, args, kwargs or {})


def wrap_engine_core_profile(
    orig: Callable[..., None],
    self: object,
    is_start: bool = True,
    profile_prefix: str | None = None,
) -> None:
    def _run() -> None:
        return orig(self, is_start, profile_prefix)

    return _run_with_engine_cpu_profile(
        self,
        is_start,
        profile_prefix,
        "EngineCore.profile",
        _run,
    )


def install() -> None:
    """Idempotent: patch vLLM :class:`EngineCore` ``collective_rpc`` / ``profile``."""
    from vllm.v1.engine.core import EngineCore

    if getattr(EngineCore, "_omni_engine_cpu_profiler_installed", False):
        return

    o_cr = EngineCore.collective_rpc
    o_pr = EngineCore.profile

    @functools.wraps(o_cr)  # type: ignore[assignment]
    def collective_rpc_patched(
        self, method: str | Any, timeout: float | None = None, args: tuple = (), kwargs: dict[str, Any] | None = None
    ) -> list[Any]:
        return wrap_engine_core_collective_rpc(o_cr, self, method, timeout, args, kwargs)

    @functools.wraps(o_pr)  # type: ignore[assignment]
    def profile_patched(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        return wrap_engine_core_profile(o_pr, self, is_start, profile_prefix)

    EngineCore.collective_rpc = collective_rpc_patched
    EngineCore.profile = profile_patched
    EngineCore._omni_engine_cpu_profiler_installed = True
    logger.info("Installed Omni EngineCore CPU profiler hooks on EngineCore.collective_rpc / .profile")

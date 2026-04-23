"""
Stage Core Process for vLLM-Omni V1 architecture.

StageEngineCoreProc inherits from vLLM's EngineCoreProc and runs the engine core
busy loop in a subprocess, communicating with StageEngineCoreClient via ZMQ.
"""

from __future__ import annotations

import os
import signal
import time
import types
from multiprocessing.process import BaseProcess
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import msgspec
import zmq
import torch
from vllm.config.profiler import _is_uri_path
from vllm.logger import init_logger
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value,
)
from vllm.utils.network_utils import get_open_zmq_ipc_path, zmq_socket_ctx
from vllm.utils.system_utils import (
    decorate_logs,
    get_mp_context,
    set_process_title,
)
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.utils import (
    EngineHandshakeMetadata,
    EngineZmqAddresses,
    get_engine_zmq_addresses,
)
from vllm.v1.utils import shutdown

from vllm_omni.engine.engine_core_cpu_profiler import install as install_engine_core_cpu_profiler_hook

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.executor import Executor

logger = init_logger(__name__)


def _resolve_engine_profiler_config(engine_core: Any) -> Any:
    """``ProfilerConfig`` is usually on ``vllm_config``; in some vLLM/omni paths the
    top-level :class:`VllmConfig` in the engine subprocess does not keep it, while
    :attr:`ModelExecutor.vllm_config` still has the same user settings as workers.
    """
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
    """Returns (prof_top, prof_cfg, tdir, tdir_is_uri, use_engine_cpu)."""
    prof_top = getattr(engine_core.vllm_config, "profiler_config", None)
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


class StageEngineCoreProc(EngineCoreProc):
    """Stage-specific engine core process for vLLM-Omni.

    Inherits from EngineCoreProc and provides its own ``run_stage_core``
    entry point for launching in a subprocess.  Does **not** delegate to
    ``EngineCoreProc.run_engine_core()``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # CPU-only Kineto trace for the scheduler + engine step (this process), see ``profile()``.
        self._engine_core_torch_profiler: torch.profiler.profile | None = None
        self._install_model_executor_profile_hook()

    @staticmethod
    def _extract_profile_control_args(
        call_args: tuple[Any, ...],
        call_kwargs: dict[str, Any],
    ) -> tuple[bool, str | None] | None:
        """Best-effort parser for model_executor.collective_rpc call signatures."""
        method = call_kwargs.get("method", call_args[0] if call_args else None)
        if method != "profile":
            return None

        rpc_args = call_kwargs.get("args", None)
        if rpc_args is None:
            # Common signature: collective_rpc(method, args, kwargs)
            if len(call_args) >= 2 and isinstance(call_args[1], (tuple, list)):
                rpc_args = call_args[1]
            else:
                # Fallback: positional tail after `method`.
                rpc_args = call_args[1:]

        if not isinstance(rpc_args, (tuple, list)) or len(rpc_args) == 0:
            return None

        is_start = bool(rpc_args[0])
        profile_prefix: str | None = rpc_args[1] if len(rpc_args) > 1 else None
        return is_start, profile_prefix

    def _install_model_executor_profile_hook(self) -> None:
        """Wrap model_executor.collective_rpc as a fallback profile interception path.

        Some vLLM versions/paths invoke model_executor.collective_rpc directly from
        the EngineCoreProc message loop, bypassing StageEngineCoreProc.collective_rpc.
        This hook guarantees scheduler/engine-core CPU profiling is toggled whenever
        profile RPCs are dispatched through model_executor.
        """
        model_executor = getattr(self, "model_executor", None)
        if model_executor is None:
            logger.warning("[engine_core profile] model_executor missing; fallback hook not installed")
            return

        original = getattr(model_executor, "collective_rpc", None)
        if not callable(original):
            logger.warning("[engine_core profile] model_executor.collective_rpc missing; fallback hook not installed")
            return

        if getattr(original, "_omni_engine_profile_wrapped", False):
            return

        def _wrapped_collective_rpc(_self: Any, *call_args: Any, **call_kwargs: Any) -> Any:
            parsed = StageEngineCoreProc._extract_profile_control_args(call_args, call_kwargs)
            if parsed is None:
                return original(*call_args, **call_kwargs)
            is_start, profile_prefix = parsed
            return self._run_with_engine_cpu_profile(
                is_start=is_start,
                profile_prefix=profile_prefix,
                from_where="model_executor.collective_rpc",
                run_workers=lambda: original(*call_args, **call_kwargs),
            )

        _wrapped_collective_rpc._omni_engine_profile_wrapped = True  # type: ignore[attr-defined]
        model_executor.collective_rpc = types.MethodType(_wrapped_collective_rpc, model_executor)
        logger.info("[engine_core profile] Installed fallback hook on model_executor.collective_rpc")

    def _log_engine_profile_path(self, where: str, is_start: bool) -> tuple[Any, bool]:
        prof_top, prof_cfg, tdir, tdir_is_uri, use_engine_cpu = _engine_profile_flags(self)
        logger.info(
            "[engine_core profile] from=%s is_start=%s stage_id=%s vllm_config.profiler_config_set=%s "
            "resolved_profiler_config_set=%s profiler=%r torch_profiler_dir_set=%s use_engine_cpu=%s",
            where,
            is_start,
            getattr(self.vllm_config.model_config, "stage_id", None),
            prof_top is not None,
            prof_cfg is not None,
            getattr(prof_cfg, "profiler", None) if prof_cfg is not None else None,
            bool(tdir) and not tdir_is_uri,
            use_engine_cpu,
        )
        return prof_cfg, use_engine_cpu

    def _run_with_engine_cpu_profile(
        self,
        is_start: bool,
        profile_prefix: str | None,
        from_where: str,
        run_workers: Callable[[], Any],
    ) -> Any:
        """``/start_profile`` / ``omni.start_profile`` use ``collective_rpc(method=\\\"profile\\\")``,
        which vLLM implements as *only* ``model_executor.collective_rpc`` and **never** calls
        :meth:`EngineCore.profile`. We must wrap the same path the API uses.
        """
        prof_cfg, use_engine_cpu = self._log_engine_profile_path(from_where, is_start)
        if is_start:
            if use_engine_cpu:
                self._start_engine_core_cpu_profiler(prof_cfg, profile_prefix)
            try:
                return run_workers()
            except Exception:
                if use_engine_cpu or self._engine_core_torch_profiler is not None:
                    self._stop_engine_core_cpu_profiler()
                raise
        else:
            out = run_workers()
            if use_engine_cpu or self._engine_core_torch_profiler is not None:
                self._stop_engine_core_cpu_profiler()
            return out

    def collective_rpc(
        self,
        method: str | Any,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[Any]:
        # Omni ``start_profile`` / ``stop_profile`` → ``collective_rpc(\\\"profile\\\", ...)`` only.
        if method == "profile" and isinstance(args, tuple) and len(args) > 0:
            is_start = bool(args[0])
            prof_prefix: str | None = args[1] if len(args) > 1 else None

            def _run() -> list[Any]:
                return super(StageEngineCoreProc, self).collective_rpc(method, timeout, args, kwargs or {})

            return self._run_with_engine_cpu_profile(is_start, prof_prefix, "collective_rpc", _run)
        return super().collective_rpc(method, timeout, args, kwargs or {})

    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        """Start/stop torch profiler in this process for scheduler (CPU) + worker (GPU) traces.

        vLLM's ``EngineCore.profile`` only forwards to ``model_executor.profile`` (workers).
        The base :class:`vllm.v1.core.sched.Scheduler` already wraps hot paths in
        ``record_function_or_nullcontext("schedule: ...")``; those only appear in a
        **torch profiler running in the EngineCore process**, not in worker traces.
        We therefore add a second profiler here with **CPU** activity when
        ``profiler_config.profiler == "torch"``.

        Note: HTTP ``/start_profile`` does **not** call this; it uses :meth:`collective_rpc` instead.
        """

        def _run() -> None:
            return super(StageEngineCoreProc, self).profile(is_start, profile_prefix)

        self._run_with_engine_cpu_profile(is_start, profile_prefix, "EngineCore.profile_utility", _run)

    def _start_engine_core_cpu_profiler(self, prof_cfg: Any, profile_prefix: str | None) -> None:
        if self._engine_core_torch_profiler is not None:
            logger.warning("Engine core torch profiler already running; skipping duplicate start")
            return
        out_dir = getattr(prof_cfg, "torch_profiler_dir", None) or ""
        if not out_dir or _is_uri_path(out_dir):
            return
        os.makedirs(out_dir, exist_ok=True)
        stage_id = getattr(self.vllm_config.model_config, "stage_id", 0)
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
        self._engine_core_cpu_trace_path = trace_path  # for logging in stop

        def on_trace_ready(prof: Any) -> None:
            try:
                prof.export_chrome_trace(trace_path)
                logger.info("Engine/scheduler trace (CPU) saved to: %s", trace_path)
            except Exception as err:
                logger.warning("Failed to export engine/scheduler trace: %s", err)

        # CPU only: workers capture CUDA in the other process. Scheduler runs here.
        self._engine_core_torch_profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=bool(getattr(prof_cfg, "torch_profiler_record_shapes", False)),
            profile_memory=bool(getattr(prof_cfg, "torch_profiler_with_memory", False)),
            with_stack=bool(getattr(prof_cfg, "torch_profiler_with_stack", False)),
            with_flops=bool(getattr(prof_cfg, "torch_profiler_with_flops", False)),
            on_trace_ready=on_trace_ready,
        )
        self._engine_core_torch_profiler.start()
        logger.info("Engine core (scheduler) torch profiler started; trace -> %s", trace_path)

    def _stop_engine_core_cpu_profiler(self) -> None:
        prof = self._engine_core_torch_profiler
        if prof is None:
            return
        try:
            prof.stop()
        except Exception as err:
            logger.warning("Error stopping engine core torch profiler: %s", err)
        finally:
            self._engine_core_torch_profiler = None

    @staticmethod
    def run_stage_core(
        *args: Any,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
        **kwargs: Any,
    ) -> None:
        """Launch StageEngineCoreProc busy loop in background process."""
        shutdown_requested = False
        maybe_register_config_serialize_by_value()
        # Ensure EngineCore.collective_rpc/profile are patched in this child
        # process even when startup bypasses package-level bootstrap hooks.
        install_engine_core_cpu_profiler_hook()

        def signal_handler(signum: int, frame: Any) -> None:
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        engine_core: StageEngineCoreProc | None = None
        try:
            vllm_config: VllmConfig = kwargs["vllm_config"]
            parallel_config = vllm_config.parallel_config

            set_process_title(f"StageEngineCoreProc_DP{dp_rank}")
            decorate_logs()

            # the current vllm-omni does not support data parallelism,
            # so we set the data parallel size to 1.
            # [TODO] support data parallelism in the future.
            # https://github.com/vllm-project/vllm-omni/issues/984
            parallel_config.data_parallel_size = 1
            parallel_config.data_parallel_size_local = 1
            parallel_config.data_parallel_rank = 0
            parallel_config.data_parallel_index = dp_rank

            engine_core = StageEngineCoreProc(
                *args,
                engine_index=dp_rank,
                **kwargs,
            )
            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("StageEngineCoreProc exiting.")
            raise
        except Exception:
            if engine_core is None:
                logger.exception("StageEngineCoreProc failed to start.")
            else:
                logger.exception("StageEngineCoreProc encountered a fatal error.")
                engine_core._send_engine_dead()
            raise
        finally:
            if engine_core is not None:
                engine_core.shutdown()


def spawn_stage_core(
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool = False,
) -> tuple[EngineZmqAddresses, BaseProcess, str]:
    """Spawn a *StageEngineCoreProc* subprocess without performing the handshake.

    Must be called while the correct device env vars are set (e.g. under
    the stage-launch lock).  Call ``complete_stage_handshake`` afterwards.

    Returns ``(addresses, process, handshake_address)``.
    """
    addresses = get_engine_zmq_addresses(vllm_config)
    handshake_address = get_open_zmq_ipc_path()

    ctx = get_mp_context()
    proc = ctx.Process(
        target=StageEngineCoreProc.run_stage_core,
        name="StageEngineCoreProc",
        kwargs={
            "vllm_config": vllm_config,
            "local_client": True,
            "handshake_address": handshake_address,
            "executor_class": executor_class,
            "log_stats": log_stats,
            "dp_rank": 0,
            "local_dp_rank": 0,
        },
    )
    proc.start()
    return addresses, proc, handshake_address


def complete_stage_handshake(
    proc: BaseProcess,
    handshake_address: str,
    addresses: EngineZmqAddresses,
    vllm_config: VllmConfig,
    handshake_timeout: int,
) -> None:
    """Perform the HELLO/INIT/READY handshake with an already-spawned proc.

    On failure the process is terminated before re-raising.
    """
    try:
        _perform_handshake(proc, handshake_address, addresses, vllm_config, handshake_timeout)
    except Exception:
        shutdown([proc])
        raise


def _perform_handshake(
    proc: BaseProcess,
    handshake_address: str,
    addresses: EngineZmqAddresses,
    vllm_config: VllmConfig,
    handshake_timeout: int,
) -> None:
    """Run the HELLO / INIT / READY handshake with the subprocess."""
    with zmq_socket_ctx(handshake_address, zmq.ROUTER, bind=True) as handshake_socket:
        poller = zmq.Poller()
        poller.register(handshake_socket, zmq.POLLIN)
        poller.register(proc.sentinel, zmq.POLLIN)

        identity, msg = _recv(poller, handshake_socket, proc, "HELLO", handshake_timeout)
        if msg.get("status") != "HELLO":
            raise RuntimeError(f"Expected HELLO, got: {msg}")

        init_payload = EngineHandshakeMetadata(
            addresses=addresses,
            parallel_config={},
        )
        handshake_socket.send_multipart([identity, msgspec.msgpack.encode(init_payload)])

        identity, msg = _recv(poller, handshake_socket, proc, "READY", handshake_timeout)
        if msg.get("status") != "READY":
            raise RuntimeError(f"Expected READY, got: {msg}")
        num_gpu_blocks = msg.get("num_gpu_blocks")
        if num_gpu_blocks is not None:
            vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks


def _recv(
    poller: zmq.Poller,
    handshake_socket: zmq.Socket,
    proc: BaseProcess,
    expected: str,
    timeout_s: int = 600,
) -> tuple[bytes, dict]:
    """Wait for one handshake message; raise if the process dies first."""
    timeout_ms = timeout_s * 1000
    while True:
        events = dict(poller.poll(timeout=timeout_ms))
        if not events:
            raise TimeoutError(
                f"Timed out waiting for {expected} from StageEngineCoreProc after {timeout_s}s. "
                f"This typically indicates model loading or initialization is taking too long. "
                f"Consider increasing `stage_init_timeout` for large models."
            )
        if handshake_socket in events:
            identity, raw = handshake_socket.recv_multipart()
            return identity, msgspec.msgpack.decode(raw)
        if proc.exitcode is not None:
            raise RuntimeError(f"StageEngineCoreProc died during {expected} (exit code {proc.exitcode})")

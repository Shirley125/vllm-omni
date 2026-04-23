"""
Stage Core Process for vLLM-Omni V1 architecture.

StageEngineCoreProc inherits from vLLM's EngineCoreProc and runs the engine core
busy loop in a subprocess, communicating with StageEngineCoreClient via ZMQ.
"""

from __future__ import annotations

import os
import signal
import time
from multiprocessing.process import BaseProcess
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

    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        """Start/stop torch profiler in this process for scheduler (CPU) + worker (GPU) traces.

        vLLM's ``EngineCore.profile`` only forwards to ``model_executor.profile`` (workers).
        The base :class:`vllm.v1.core.sched.Scheduler` already wraps hot paths in
        ``record_function_or_nullcontext("schedule: ...")``; those only appear in a
        **torch profiler running in the EngineCore process**, not in worker traces.
        We therefore add a second profiler here with **CPU** activity when
        ``profiler_config.profiler == "torch"``.
        """
        prof_top = getattr(self.vllm_config, "profiler_config", None)
        prof_cfg = _resolve_engine_profiler_config(self)
        tdir = getattr(prof_cfg, "torch_profiler_dir", None) or "" if prof_cfg is not None else ""
        tdir_is_uri = bool(tdir) and _is_uri_path(tdir)
        use_engine_cpu = (
            prof_cfg is not None
            and getattr(prof_cfg, "profiler", None) == "torch"
            and bool(tdir)
            and not tdir_is_uri
        )
        # Always one line: otherwise operators see *no* StageEngineCoreProc log when
        # the engine snapshot lacks profiler_config (very common) or URI dirs skip CPU trace.
        logger.info(
            "[engine_core profile] is_start=%s stage_id=%s vllm_config.profiler_config_set=%s "
            "resolved_profiler_config_set=%s profiler=%r torch_profiler_dir_set=%s "
            "use_engine_cpu=%s",
            is_start,
            getattr(self.vllm_config.model_config, "stage_id", None),
            prof_top is not None,
            prof_cfg is not None,
            getattr(prof_cfg, "profiler", None) if prof_cfg is not None else None,
            bool(tdir) and not tdir_is_uri,
            use_engine_cpu,
        )
        if is_start:
            if use_engine_cpu:
                self._start_engine_core_cpu_profiler(prof_cfg, profile_prefix)
            try:
                super().profile(is_start, profile_prefix)
            except Exception:
                if use_engine_cpu or self._engine_core_torch_profiler is not None:
                    self._stop_engine_core_cpu_profiler()
                raise
        else:
            super().profile(is_start, profile_prefix)
            if use_engine_cpu or self._engine_core_torch_profiler is not None:
                self._stop_engine_core_cpu_profiler()

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

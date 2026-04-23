"""Process-wide bootstrap for omni profiling hooks.

Python imports ``sitecustomize`` automatically during interpreter startup
when the module is importable on ``sys.path``. This makes it the safest
place to install hooks that must also apply to subprocesses created by
CoreEngineProcManager.
"""

from __future__ import annotations

try:
    from vllm_omni.engine.engine_core_cpu_profiler import install as _install_engine_cpu_profiler_hook

    _install_engine_cpu_profiler_hook()
except Exception:
    # Best effort only: never break process startup if hook install fails.
    pass

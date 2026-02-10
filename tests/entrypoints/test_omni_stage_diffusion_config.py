# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.entrypoints.omni_stage import _build_od_config, _resolve_worker_cls

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_build_od_config_includes_diffusion_fields():
    engine_args = {
        "cache_backend": "cache_dit",
        "cache_config": {"Fn_compute_blocks": 2},
        "vae_use_slicing": True,
    }
    od_config = _build_od_config(engine_args, model="dummy-model")

    assert od_config["model"] == "dummy-model"
    assert od_config["cache_backend"] == "cache_dit"
    assert od_config["cache_config"]["Fn_compute_blocks"] == 2
    assert od_config["vae_use_slicing"] is True


def test_build_od_config_respects_explicit_config():
    engine_args = {
        "od_config": {"cache_backend": "tea_cache"},
        "cache_backend": "cache_dit",
    }
    od_config = _build_od_config(engine_args, model="dummy-model")
    assert od_config == {"cache_backend": "tea_cache"}


@pytest.mark.parametrize(
    ("worker_type", "expected_cls"),
    [("ar", "ARWorker"), ("generation", "GENWorker")],
)
def test_resolve_worker_cls_sets_worker_class_and_keeps_worker_type(monkeypatch, worker_type, expected_cls):
    class _FakePlatform:
        @staticmethod
        def get_omni_ar_worker_cls():
            return "ARWorker"

        @staticmethod
        def get_omni_generation_worker_cls():
            return "GENWorker"

    monkeypatch.setattr("vllm_omni.platforms.current_omni_platform", _FakePlatform())

    engine_args = {"worker_type": worker_type}
    _resolve_worker_cls(engine_args)

    assert engine_args["worker_cls"] == expected_cls
    assert engine_args["worker_type"] == worker_type


def test_resolve_worker_cls_raises_on_unknown_type():
    engine_args = {"worker_type": "unknown_worker_type"}
    with pytest.raises(ValueError, match="Unknown worker_type"):
        _resolve_worker_cls(engine_args)

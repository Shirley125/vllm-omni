# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for VoxCPM2 talker per-request state lifecycle."""

from __future__ import annotations

import functools

import pytest

torch = pytest.importorskip("torch")


@functools.lru_cache(maxsize=1)
def _voxcpm2_talker_mod():
    """Defer talker import (pulls vLLM model_executor) until first use."""
    from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import (
        VoxCPM2TalkerForConditionalGeneration,
        _RequestState,
    )

    return VoxCPM2TalkerForConditionalGeneration, _RequestState


def _make_bare_talker():
    VoxCPM2TalkerForConditionalGeneration, _ = _voxcpm2_talker_mod()
    talker = VoxCPM2TalkerForConditionalGeneration.__new__(VoxCPM2TalkerForConditionalGeneration)
    talker._active_states = {}
    talker._current_request_id = None
    talker._pending_requests = []
    talker._results_queue = []
    talker._audio_queue = []
    talker._deferred_cleanup_ids = set()
    talker._max_batch_size = 4
    talker._active_state_warn_threshold = 512
    talker._active_state_warned = False
    return talker


def _seed_cached_decode(talker, req_id: str):
    _, _RequestState = _voxcpm2_talker_mod()
    state = _RequestState(request_id=req_id)
    state.prefill_completed = True
    state.decode_step_count = 5
    talker._active_states[req_id] = state
    return state


class TestStateEvictionContract:
    def test_pending_requests_is_not_used_for_eviction(self) -> None:
        talker = _make_bare_talker()

        cached_ids = [f"req-{i}" for i in range(4)]
        for rid in cached_ids:
            _seed_cached_decode(talker, rid)

        walked_so_far = ["req-new", cached_ids[0], cached_ids[1]]
        talker._pending_requests = [(rid, False, None, 0) for rid in walked_so_far]

        for rid in cached_ids:
            assert rid in talker._active_states
            assert talker._active_states[rid].prefill_completed is True

    def test_on_requests_finished_defers_cleanup(self) -> None:
        talker = _make_bare_talker()
        _seed_cached_decode(talker, "req-A")
        _seed_cached_decode(talker, "req-B")

        talker.on_requests_finished({"req-A"})

        assert "req-A" in talker._active_states
        assert "req-A" in talker._deferred_cleanup_ids

    def test_flush_deferred_cleanup_removes_only_finished(self) -> None:
        talker = _make_bare_talker()
        _seed_cached_decode(talker, "req-A")
        _seed_cached_decode(talker, "req-B")
        talker.on_requests_finished(["req-A"])

        talker._flush_deferred_cleanup()

        assert "req-A" not in talker._active_states
        assert "req-B" in talker._active_states
        assert talker._deferred_cleanup_ids == set()

    def test_current_request_id_cleared_when_matching(self) -> None:
        talker = _make_bare_talker()
        _seed_cached_decode(talker, "req-A")
        talker._current_request_id = "req-A"

        talker.on_requests_finished({"req-A"})
        talker._flush_deferred_cleanup()

        assert talker._current_request_id is None

    def test_current_request_id_preserved_when_not_finished(self) -> None:
        talker = _make_bare_talker()
        _seed_cached_decode(talker, "req-A")
        _seed_cached_decode(talker, "req-B")
        talker._current_request_id = "req-B"

        talker.on_requests_finished({"req-A"})
        talker._flush_deferred_cleanup()

        assert talker._current_request_id == "req-B"


class TestLeakWarnGuard:
    def test_warn_fires_once_over_threshold(self, monkeypatch) -> None:
        from vllm_omni.model_executor.models.voxcpm2 import voxcpm2_talker as tk

        calls: list[str] = []

        def _capture(msg, *args, **kwargs):
            calls.append(msg % args if args else msg)

        monkeypatch.setattr(tk.logger, "warning", _capture)

        talker = _make_bare_talker()
        talker._active_state_warn_threshold = 3

        _, RState = _voxcpm2_talker_mod()
        for i in range(4):
            talker._active_states[f"seed-{i}"] = RState(request_id=f"seed-{i}")

        talker._get_or_create_state("new-1")
        talker._get_or_create_state("new-2")

        leak_warnings = [m for m in calls if "cleanup path leak" in m]
        assert len(leak_warnings) == 1
        assert talker._active_state_warned is True


class _NoopPerf:
    def start(self, name: str) -> None:
        pass

    def stop(self, name: str) -> None:
        pass


class _FakeTTS:
    def feat_encoder(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def enc_to_lm_proj(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(1).mean(dim=1).repeat(1, 4)


class TestDecodeBatchContract:
    def _make_decode_talker(self) -> object:
        talker = _make_bare_talker()
        talker._perf = _NoopPerf()
        talker._tts = _FakeTTS()
        talker._side_dtype = torch.float32
        talker._patch_size = 2
        talker._feat_dim = 3

        def dit_proj(lm_h: torch.Tensor, res_h: torch.Tensor) -> torch.Tensor:
            return lm_h + 2 * res_h

        def stop_fn(lm_h: torch.Tensor) -> torch.Tensor:
            return torch.stack([lm_h.sum(dim=1), -lm_h.sum(dim=1)], dim=1)

        def run_cfm(dit_h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
            return dit_h[:, :1].view(-1, 1, 1) + cond.transpose(1, 2) * 0.1

        talker._dit_proj_fn = dit_proj
        talker._stop_fn = stop_fn
        talker._run_cfm = run_cfm
        return talker

    def _make_decode_state(self, request_id: str, offset: float):
        _, RState = _voxcpm2_talker_mod()
        state = RState(request_id=request_id)
        state.curr_prefix_feat_cond = torch.arange(6, dtype=torch.float32).reshape(2, 3) + offset
        return state

    def test_batch_decode_matches_per_request_state_updates(self) -> None:
        seq_talker = self._make_decode_talker()
        batch_talker = self._make_decode_talker()

        seq_states = [self._make_decode_state("req-0", 0.0), self._make_decode_state("req-1", 10.0)]
        batch_states = [self._make_decode_state("req-0", 0.0), self._make_decode_state("req-1", 10.0)]
        metas = [
            {"new_lm_hidden": torch.tensor([[1.0, 2.0, 3.0, 4.0]])},
            {"new_lm_hidden": torch.tensor([[5.0, 6.0, 7.0, 8.0]])},
        ]
        batch_out = torch.tensor([[0.5, 1.0, 1.5, 2.0], [2.5, 3.0, 3.5, 4.0]])

        for state, meta, res_out in zip(seq_states, metas, batch_out, strict=True):
            seq_talker._finish_decode(state, meta, res_out.unsqueeze(0), torch.device("cpu"))

        batch_talker._finish_decode_batch(
            [(state, False, meta) for state, meta in zip(batch_states, metas, strict=True)],
            batch_out,
        )

        for expected, actual in zip(seq_states, batch_states, strict=True):
            torch.testing.assert_close(actual.precomputed_stop_logits, expected.precomputed_stop_logits)
            torch.testing.assert_close(actual.curr_embed_for_next, expected.curr_embed_for_next)
            torch.testing.assert_close(actual.prev_feat_embed, expected.prev_feat_embed)
            torch.testing.assert_close(actual.curr_prefix_feat_cond, expected.curr_prefix_feat_cond)
            torch.testing.assert_close(actual.last_audio_patch_gpu, expected.last_audio_patch_gpu)

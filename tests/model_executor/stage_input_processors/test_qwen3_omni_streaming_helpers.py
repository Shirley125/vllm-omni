# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Qwen3-Omni streaming thinker→talker / talker→codec helpers (PR #2581)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import vllm_omni.model_executor.stage_input_processors.qwen3_omni as q3

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _clear_qwen3_streaming_globals() -> None:
    q3._THINKER2TALKER_STREAMING_STATE.clear()
    q3._TALKER2CODE2WAV_LAST_SEQ_LEN.clear()
    yield
    q3._THINKER2TALKER_STREAMING_STATE.clear()
    q3._TALKER2CODE2WAV_LAST_SEQ_LEN.clear()


def test_get_streaming_talker_tokens_first_segment() -> None:
    inc_p, inc_o, merged, thinker_in = q3._get_streaming_talker_tokens(
        "r1",
        [1, 2],
        [10, 11],
    )
    assert inc_p == [1, 2]
    assert inc_o == [10, 11]
    assert merged == [1, 2, 10, 11]
    assert thinker_in == [1, 2]


def test_get_streaming_talker_tokens_second_segment_accumulates() -> None:
    q3._get_streaming_talker_tokens("r2", [1, 2], [10, 11])
    inc_p, inc_o, merged, thinker_in = q3._get_streaming_talker_tokens("r2", [1, 2, 3, 4], [10, 11, 12, 13])
    assert inc_p == [3, 4]
    assert inc_o == [12, 13]
    assert merged == [1, 2, 10, 3, 4, 12, 13]
    assert thinker_in == [1, 2, 10, 3, 4]


def test_get_streaming_talker_tokens_new_prompt_len_snapshot_truncates() -> None:
    inc_p, inc_o, merged, thinker_in = q3._get_streaming_talker_tokens(
        "r3",
        [1, 2, 3, 4, 5, 6],
        [10],
        new_prompt_len_snapshot=2,
    )
    assert inc_p == [1, 2, 3, 4]
    assert inc_o == [10]
    assert merged == [1, 2, 3, 4, 10]
    assert thinker_in == [1, 2, 3, 4]


def test_get_streaming_talker_tokens_clear_state() -> None:
    q3._get_streaming_talker_tokens("r4", [1], [2], clear_state=True)
    assert "r4" not in q3._THINKER2TALKER_STREAMING_STATE


def test_get_streaming_codec_delta_len_increments_and_finishes() -> None:
    d1 = q3._get_streaming_codec_delta_len(5, "c1", SimpleNamespace(finished=False))
    assert d1 == 5
    d2 = q3._get_streaming_codec_delta_len(8, "c1", SimpleNamespace(finished=False))
    assert d2 == 2
    # After d2, stored cursor is cur_seq_len + 1 == 9; next delta uses new cur_seq_len - 9.
    d3 = q3._get_streaming_codec_delta_len(10, "c1", SimpleNamespace(finished=True))
    assert d3 == 1
    assert "c1" not in q3._TALKER2CODE2WAV_LAST_SEQ_LEN

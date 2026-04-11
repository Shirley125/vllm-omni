# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
"""Stage input processor for Qwen3 Omni MoE: Thinker → Talker transition."""

import os
from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.platforms import current_platform

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.stage_input_processors.tts_utils import (
    extract_language_from_prompt,
    extract_language_from_request,
    extract_speaker_from_prompt,
    extract_speaker_from_request,
)

_THINKER2TALKER_LAST_PROMPT_LEN: dict[str, int] = {}
_THINKER2TALKER_LAST_OUTPUT_LEN: dict[str, int] = {}
_THINKER2TALKER_STREAMING_TOKEN_CACHE: dict[str, dict[str, list[int]]] = {}


def _compute_talker_prompt_ids_length(info, device: torch.device | str = "cuda") -> int:
    im_start_token_id = 151644
    system_token_id = 8948
    user_token_id = 872
    assistant_token_id = 77091

    thinker_sequences = torch.tensor(info["thinker_sequences"], dtype=torch.long, device=device).unsqueeze(0)  # [1, T]

    input_ids = torch.tensor(info["thinker_input_ids"], dtype=torch.long, device=device).unsqueeze(0)  # [1, T]

    im_start_indexes = torch.cat(
        [
            torch.nonzero(input_ids[0] == im_start_token_id).squeeze(1),
            torch.tensor([thinker_sequences.shape[-1]], device=input_ids.device, dtype=input_ids.dtype),
        ],
        dim=0,
    )

    sum_user_len = 0
    assistant_len = 0
    for i in range(len(im_start_indexes) - 1):
        s = int(im_start_indexes[i].item())
        e = int(im_start_indexes[i + 1].item())
        role = int(input_ids[0, s + 1].item())
        if role == system_token_id:
            continue
        elif role == user_token_id:
            sum_user_len += e - s
        elif role == assistant_token_id and i == len(im_start_indexes) - 2:
            assistant_len += 9  # 3 + 4 + 1 + 1
        else:
            pass
    print(f"cwj thinker sum_user_len = {sum_user_len}, assistant_len = {assistant_len}")
    return sum_user_len + assistant_len


# =========================
# Common helpers
# =========================


def _ensure_list(x):
    """Convert ConstantList / tensor-like to Python list."""
    if hasattr(x, "_x"):
        return list(x._x)
    elif not isinstance(x, list):
        return x
    return list(x)


def _validate_stage_inputs(stage_list, engine_input_source):
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    stage_id = engine_input_source[0]
    if stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {stage_id}")

    stage = stage_list[stage_id]
    if stage.engine_outputs is None:
        raise RuntimeError(f"Stage {stage_id} has no outputs yet")

    return stage.engine_outputs


def _trim_leading_zero_rows(code_rows: torch.Tensor) -> torch.Tensor:
    """Drop prefill placeholder rows that are all zeros.

    In talker stage, prefill may initialize code_predictor_codes with zero rows.
    For code2wav, we should keep only rows from the first non-zero row onward.
    """
    if not isinstance(code_rows, torch.Tensor) or code_rows.numel() == 0:
        return code_rows
    if code_rows.ndim != 2:
        return code_rows
    non_zero_rows = (code_rows != 0).any(dim=1)
    if not bool(non_zero_rows.any().item()):
        return code_rows[:0]
    first_valid = int(torch.nonzero(non_zero_rows, as_tuple=False)[0].item())
    return code_rows[first_valid:]


def _slice_incremental_tokens(
    request_id: str,
    prompt_token_ids: list[int],
    output_token_ids: list[int],
    *,
    clear_state: bool = False,
) -> tuple[list[int], list[int]]:
    """Return incremental prompt/output token slices for one request."""
    prev_prompt_len = _THINKER2TALKER_LAST_PROMPT_LEN.get(request_id, 0)
    prev_output_len = _THINKER2TALKER_LAST_OUTPUT_LEN.get(request_id, 0)

    cur_prompt_len = len(prompt_token_ids)
    cur_output_len = len(output_token_ids)

    inc_prompt = prompt_token_ids[prev_prompt_len:]
    inc_output = output_token_ids[prev_output_len:]

    _THINKER2TALKER_LAST_PROMPT_LEN[request_id] = cur_prompt_len
    _THINKER2TALKER_LAST_OUTPUT_LEN[request_id] = cur_output_len

    if clear_state:
        _THINKER2TALKER_LAST_PROMPT_LEN.pop(request_id, None)
        _THINKER2TALKER_LAST_OUTPUT_LEN.pop(request_id, None)

    return inc_prompt, inc_output


def _merge_streaming_token_info(
    request_id: str,
    prompt_token_ids: list[int],
    output_token_ids: list[int],
    *,
    clear_state: bool = False,
) -> tuple[list[int], list[int]]:
    """Merge token deltas and derive thinker_input_ids from thinker_sequences."""
    merged_seq = prompt_token_ids + output_token_ids
    if not merged_seq:
        if clear_state:
            _THINKER2TALKER_STREAMING_TOKEN_CACHE.pop(request_id, None)
        return merged_seq, []

    cached = _THINKER2TALKER_STREAMING_TOKEN_CACHE.get(request_id)
    old_seq = []
    if cached is not None:
        old_seq = cached.get("thinker_sequences", [])
        if old_seq is not None:
            merged_seq = old_seq + merged_seq

    merged_input_ids = old_seq + prompt_token_ids
    _THINKER2TALKER_STREAMING_TOKEN_CACHE[request_id] = {
        "thinker_sequences": merged_seq[:-1],
        "thinker_input_ids": merged_input_ids,
    }

    if clear_state:
        _THINKER2TALKER_STREAMING_TOKEN_CACHE.pop(request_id, None)

    return merged_seq, merged_input_ids


# =========================
# Thinker -> Talker
# =========================


def thinker2talker_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
    is_segment_finished: bool = False,
) -> list[dict[str, Any]]:
    """
    Process thinker outputs to create talker inputs.
    1. thinker's text generation outputs (token IDs + hidden states)
    2. Split hidden states into: prompt embeddings + generated embeddings
    3. Package for talker with additional information
    """

    request_id = request.external_req_id
    chunk_id = transfer_manager.put_req_chunk[request_id]
    if chunk_id == 0:
        all_token_ids = request.all_token_ids  # prefill + decode
        prompt_token_ids = request.prompt_token_ids
        # Convert ConstantList to regular list for OmniSerializer serialization
        all_token_ids = _ensure_list(all_token_ids)
        prompt_token_ids = _ensure_list(prompt_token_ids)
        talker_additional_info = {
            "thinker_prefill_embeddings": pooling_output.get("0").detach().cpu(),
            "thinker_hidden_states": pooling_output.get("24").detach().cpu(),
            "thinker_sequences": all_token_ids,
            "thinker_input_ids": prompt_token_ids,
            # Provide thinker-side TTS token embeddings for talker projection
            "tts_bos_embed": pooling_output.get("tts_bos_embed").detach().cpu(),
            "tts_eos_embed": pooling_output.get("tts_eos_embed").detach().cpu(),
            "tts_pad_embed": pooling_output.get("tts_pad_embed").detach().cpu(),
            "finished": torch.tensor(is_finished, dtype=torch.bool),
            "is_segment_finished": torch.tensor(is_segment_finished, dtype=torch.bool),
        }
        speaker = extract_speaker_from_request(request)
        if speaker is not None:
            talker_additional_info["speaker"] = speaker
        language = extract_language_from_request(request)
        if language is not None:
            talker_additional_info["language"] = language
        if transfer_manager.request_payload.get(request_id) is None:
            if not is_finished:
                transfer_manager.request_payload[request_id] = talker_additional_info
                return None
        else:
            save_payload = transfer_manager.request_payload.pop(request_id)
            talker_additional_info["thinker_prefill_embeddings"] = torch.cat(
                (
                    save_payload.get("thinker_prefill_embeddings"),
                    talker_additional_info.get("thinker_prefill_embeddings"),
                ),
                dim=0,
            )
            talker_additional_info["thinker_hidden_states"] = torch.cat(
                (save_payload.get("thinker_hidden_states"), talker_additional_info.get("thinker_hidden_states")),
                dim=0,
            )
    else:
        output_token_ids = request.output_token_ids
        # Convert ConstantList to regular list for OmniSerializer serialization
        output_token_ids = _ensure_list(output_token_ids)

        talker_additional_info = {
            "finished": torch.tensor(is_finished, dtype=torch.bool),
            "is_segment_finished": torch.tensor(is_segment_finished, dtype=torch.bool),
        }
        speaker = extract_speaker_from_request(request)
        if speaker is not None:
            talker_additional_info["speaker"] = speaker
        language = extract_language_from_request(request)
        if language is not None:
            talker_additional_info["language"] = language

        if output_token_ids:
            talker_additional_info["override_keys"] = ["thinker_decode_embeddings", "thinker_output_token_ids"]
            talker_additional_info["thinker_decode_embeddings"] = pooling_output.get("0").detach().cpu()
            talker_additional_info["thinker_output_token_ids"] = output_token_ids
        else:
            # When prefilling a chunked thinker, thinker_hidden_states needs to be updated.
            talker_additional_info["thinker_prefill_embeddings"] = pooling_output.get("0").detach().cpu()
            talker_additional_info["thinker_hidden_states"] = pooling_output.get("24").detach().cpu()

    return talker_additional_info

def thinker2talker(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
    new_prompt_len_snapshot: Any | None = None,
    is_streaming_session: bool = False,
) -> list[OmniTokensPrompt]:
    """
    Process thinker outputs to create talker inputs.

    Workflow:
    1. Extract thinker's text generation outputs (token IDs + hidden states)
    2. Split hidden states into: prompt embeddings + generated embeddings
    3. Package for talker with additional information

    Args:
        stage_list: List of stage objects
        engine_input_source: Source stage IDs (typically [0] for thinker)
        prompt: Original prompt data
        requires_multimodal_data: Whether multimodal data is required

    Returns:
        List of OmniTokensPrompt for talker stage
    """
    thinker_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    talker_inputs: list[OmniTokensPrompt] = []

    device = torch.device(current_platform.device_type)

    # Process each thinker output
    for i, thinker_output in enumerate(thinker_outputs):
        output = thinker_output.outputs[0]
        req_id = str(getattr(thinker_output, "request_id", f"idx-{i}"))
        # todo: The next streaming segment is already concatenated to the prompt,
        #  so it should be truncated, except last segment
        print(f"cwj thinker2talker input ids = {thinker_output.prompt_token_ids}, new_prompt_len_snapshot = {new_prompt_len_snapshot}")
        print(f"cwj thinker2talker output ids = {output.token_ids}")
        prompt_token_ids = thinker_output.prompt_token_ids
        output_ids = output.token_ids
        if is_streaming_session:
            if new_prompt_len_snapshot:
                prompt_token_ids = thinker_output.prompt_token_ids[:-new_prompt_len_snapshot]
            prompt_token_ids, output_ids = _slice_incremental_tokens(
                req_id,
                prompt_token_ids,
                output_ids,
                clear_state=bool(getattr(thinker_output, "finished", False)),
            )
            thinker_sequences, thinker_input_ids = _merge_streaming_token_info(
                req_id,
                prompt_token_ids,
                output_ids,
                clear_state=bool(getattr(thinker_output, "finished", False)),
            )
        else:
            thinker_sequences = prompt_token_ids + output_ids
            thinker_input_ids = prompt_token_ids
        incremental_cached_token_length = len(prompt_token_ids + output_ids) - 1

        info = {
            "thinker_prefill_embeddings": output.multimodal_output["0"].detach().to(device=device, dtype=torch.float)[-incremental_cached_token_length:],
            "thinker_hidden_states": output.multimodal_output["24"].detach().to(device=device, dtype=torch.float)[-incremental_cached_token_length:],
            "thinker_sequences": thinker_sequences,
            "thinker_input_ids": thinker_input_ids,
            # Provide thinker-side TTS token embeddings for talker projection
            "tts_bos_embed": output.multimodal_output["tts_bos_embed"].detach().to(device=device, dtype=torch.float),
            "tts_eos_embed": output.multimodal_output["tts_eos_embed"].detach().to(device=device, dtype=torch.float),
            "tts_pad_embed": output.multimodal_output["tts_pad_embed"].detach().to(device=device, dtype=torch.float),
        }
        speaker = extract_speaker_from_prompt(prompt, index=i)
        if speaker is not None:
            info["speaker"] = speaker
        language = extract_language_from_prompt(prompt, index=i)
        if language is not None:
            info["language"] = language

        # print(f"cwj input process len(thinker_sequences) = {len(info.get('thinker_sequences'))}")
        print(f"cwj input process len(thinker_sequences) = {len(thinker_sequences)}")
        print(f"cwj input process len(thinker_input_ids) = {len(thinker_input_ids)}")
        print(f"cwj input process thinker_hidden.shape[0] = {info.get('thinker_hidden_states').shape[0]}")
        print(f"cwj input process thinker_prefill_embeddings.shape[0] = {info.get('thinker_prefill_embeddings').shape[0]}")

        prompt_len = _compute_talker_prompt_ids_length(info, device=device)
        print(f"cwj thinker prompt len: {len(thinker_input_ids)}, sequences len: {len(thinker_sequences)}, "
              f"talker prompt len: {prompt_len}")
        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * prompt_len,
                additional_information=info,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return talker_inputs


# =========================
# Talker -> Code2Wav
# =========================


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
    is_segment_finished: bool = False,
):
    """
    Pooling version.
    """
    if "code_predictor_codes" not in pooling_output:
        return None

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size_config = int(cfg.get("codec_chunk_frames", 25))
    left_context_size_config = int(cfg.get("codec_left_context_frames", 25))

    code_predictor_codes = pooling_output["code_predictor_codes"]

    if code_predictor_codes is None:
        return None
    if isinstance(code_predictor_codes, torch.Tensor):
        if code_predictor_codes.numel() == 0:
            return None
    elif hasattr(code_predictor_codes, "__len__"):
        if len(code_predictor_codes) == 0:
            return None

    if isinstance(code_predictor_codes, torch.Tensor):
        if not code_predictor_codes.any():
            return None
    else:
        code_tensor = torch.tensor(code_predictor_codes, dtype=torch.long)
        if not code_tensor.any():
            return None

    codec_codes = code_predictor_codes.to(torch.long).transpose(0, 1).cpu().to(torch.long).reshape(-1).tolist()
    if sum(codec_codes) == 0:
        return None

    request_id = request.external_req_id
    transfer_manager.code_prompt_token_ids[request_id].append(codec_codes)
    length = len(transfer_manager.code_prompt_token_ids[request_id])
    chunk_length = length % chunk_size_config
    if chunk_length != 0 and not is_finished:
        return None

    context_length = chunk_length if chunk_length != 0 else chunk_size_config
    # ensure left context does not exceed available length
    left_context_size = max(0, min(length - context_length, left_context_size_config))
    end_index = min(length, left_context_size + context_length)

    codes = (
        torch.tensor(transfer_manager.code_prompt_token_ids[request_id][-end_index:])
        .transpose(0, 1)
        .reshape(-1)
        .tolist()
    )

    info = {
        "code_predictor_codes": codes,
        "left_context_size": left_context_size,
        "finished": torch.tensor(is_finished, dtype=torch.bool),
        "is_segment_finished": torch.tensor(is_segment_finished, dtype=torch.bool),
    }
    return info


def talker2code2wav(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
    new_prompt_len_snapshot: Any | None = None,
    is_streaming_session: bool = False,
) -> list[OmniTokensPrompt]:
    """
    Process talker outputs to create code2wav inputs.

    Workflow:
    1. Extract talker's codec code outputs (8-layer RVQ codes)
    2. Flatten codes for code2wav input
    3. Package for code2wav stage

    Args:
        stage_list: List of stage objects
        engine_input_source: Source stage IDs (typically [1] for talker)
        prompt: Original prompt data
        requires_multimodal_data: Whether multimodal data is required

    Returns:
        List of OmniTokensPrompt for code2wav stage
    """
    talker_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    code2wav_inputs: list[OmniTokensPrompt] = []
    # Process each talker output
    for talker_output in talker_outputs:
        output = talker_output.outputs[0]
        seq_len = len(output.token_ids) - 1
        print(f"cwj talker2code2wav output = {output}, seq_len = {seq_len}")
        # Extract codec codes from talker output
        # Expected shape: [8, seq_len] (8-layer RVQ codes)
        code_rows = output.multimodal_output["code_predictor_codes"][-seq_len:].to(torch.long)
        code_rows = _trim_leading_zero_rows(code_rows)
        codec_codes = code_rows.transpose(0, 1).cpu().reshape(-1).tolist()  # 16, seq_len_eff
        print(f"cwj talker2code2wav codec_codes: {codec_codes}")
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return code2wav_inputs

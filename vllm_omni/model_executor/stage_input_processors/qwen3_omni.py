# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
"""Stage input processor for Qwen3 Omni MoE: Thinker → Talker transition."""

from typing import Any

import torch

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.inputs.data import OmniTokensPrompt


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

    return sum_user_len + assistant_len


def thinker2talker(
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
) -> list[dict[str, Any]]:
    """
    Process thinker outputs to create talker inputs.
    1. thinker's text generation outputs (token IDs + hidden states)
    2. Split hidden states into: prompt embeddings + generated embeddings
    3. Package for talker with additional information
    """
    all_token_ids = request.all_token_ids  # prefill + decode
    prompt_token_ids = request.prompt_token_ids

    # Convert ConstantList to regular list for OmniSerializer serialization
    if hasattr(all_token_ids, "_x"):
        all_token_ids = list(all_token_ids._x)
    elif not isinstance(all_token_ids, list):
        all_token_ids = list(all_token_ids)

    if hasattr(prompt_token_ids, "_x"):
        prompt_token_ids = list(prompt_token_ids._x)
    elif not isinstance(prompt_token_ids, list):
        prompt_token_ids = list(prompt_token_ids)

    thinker_output = pooling_output

    # print(f"thinker_outputs: {thinker_output}")

    talker_additional_info = {
        "thinker_embeddings": thinker_output.get("0").detach().cpu(),
        "thinker_hidden_states": thinker_output.get("24").detach().cpu(),
        "thinker_sequences": all_token_ids,
        "thinker_input_ids": prompt_token_ids,
        # Provide thinker-side TTS token embeddings for talker projection
        "tts_bos_embed": thinker_output.get("tts_bos_embed")[0].detach().cpu(),
        "tts_eos_embed": thinker_output.get("tts_eos_embed")[0].detach().cpu(),
        "tts_pad_embed": thinker_output.get("tts_pad_embed")[0].detach().cpu(),
        "finished": torch.tensor(request.is_finished(), dtype=torch.bool),
    }
    #     OmniTokensPrompt(
    #     prompt_token_ids=[0] * _compute_talker_prompt_ids_length(info),
    #     additional_information=info,
    #     multi_modal_data=None,
    #     mm_processor_kwargs=None,
    # )
    # )
    # print(f"talker_inputs: {talker_inputs}")

    return talker_additional_info


def talker2code2wav(
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
) -> list[OmniTokensPrompt]:
    """
    Process talker outputs to create code2wav inputs.
    1. Check if talker has generated first codebook (prefill complete)
    2. Extract talker's codec code outputs
    3. Flatten codes for code2wav input
    4. Package for code2wav stage

    Args:
        stage_list: List of stage objects
        engine_input_source: Source stage IDs (typically [1] for talker)
        prompt: Original prompt data
        requires_multimodal_data: Whether multimodal data is required

    Returns:
        List of OmniTokensPrompt for code2wav stage

    Note:
        Returns empty list if codebook is not yet generated (prefill not complete)
    """
    talker_output = pooling_output
    if "code_predictor_codes" not in talker_output:
        return []

    code_predictor_codes = talker_output["code_predictor_codes"]  # (num_code_groups, num_codes)

    if code_predictor_codes is None:
        return []
    if isinstance(code_predictor_codes, torch.Tensor):
        if code_predictor_codes.shape[0] == 0:
            return []
    elif hasattr(code_predictor_codes, "__len__"):
        if len(code_predictor_codes) == 0:
            return []

    codec_codes = (
        code_predictor_codes.to(torch.long).transpose(0, 1).cpu().to(torch.long).reshape(-1).tolist()
    )  # 16, seq_len

    # code = torch.flatten(pooling_output["code_predictor_codes"]).detach().cpu()
    code2wav_additional_info = {
        "code_predictor_codes": codec_codes,
        "finished": torch.tensor(request.is_finished(), dtype=torch.bool),
    }
    return code2wav_additional_info

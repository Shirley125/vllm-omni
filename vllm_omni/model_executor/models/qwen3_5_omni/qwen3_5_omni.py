# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
"""Inference-only Qwen3-Omni-Moe unified model (thinker + talker + audio_tokenizer)."""

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.models.interfaces import IsHybrid, SupportsMRoPE, SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.qwen3_5_omni.qwen3_5_omni_thinker import (
    Qwen3_5OmniConditionalGenerationMixin,
    Qwen3_5OmniThinkerDummyInputsBuilder,
    Qwen3_5OmniThinkerForConditionalGeneration,
    Qwen3_5OmniThinkerMultiModalProcessor,
    Qwen3_5OmniThinkerProcessingInfo,
)
from vllm_omni.model_executor.models.utils import add_prefix_to_loaded_weights
from vllm_omni.transformers_utils.configs.configuration_qwen3_5_omni import (
    Qwen3_5OmniAudioTokenizerConfig,
    Qwen3_5OmniConfig,
    Qwen3_5OmniTalkerConfig,
    Qwen3_5OmniThinkerConfig,
)

logger = init_logger(__name__)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3_5OmniThinkerMultiModalProcessor,
    info=Qwen3_5OmniThinkerProcessingInfo,
    dummy_inputs=Qwen3_5OmniThinkerDummyInputsBuilder,
)
class Qwen3_5OmniForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    Qwen3_5OmniConditionalGenerationMixin,
    CustomProcessMixin,
    SupportsMRoPE,
    IsHybrid,
):
    config_class = Qwen3_5OmniConfig

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        return Qwen3_5OmniThinkerForConditionalGeneration.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        return Qwen3_5OmniThinkerForConditionalGeneration.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        return Qwen3_5OmniThinkerForConditionalGeneration.get_mamba_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        # Keep vllm_config for later submodule init
        config: Qwen3_5OmniConfig = vllm_config.model_config.hf_config
        self.multimodal_config = vllm_config.model_config.multimodal_config

        # Keep vllm_config for later submodule init
        self.vllm_config = vllm_config
        self.config = config

        # Initialize thinker components
        self.thinker_config: Qwen3_5OmniThinkerConfig = config.thinker_config

        # Initialize talker components
        self.talker_config: Qwen3_5OmniTalkerConfig = config.talker_config

        # Initialize code2wav components
        self.audio_tokenizer_config: Qwen3_5OmniAudioTokenizerConfig = config.audio_tokenizer_config
        self.quant_config = vllm_config.quant_config

        # Determine model stage
        self.model_stage = vllm_config.model_config.model_stage
        if self.model_stage == "thinker":
            thinker_vllm_config = vllm_config.with_hf_config(
                self.thinker_config, architectures=["Qwen3_5OmniThinkerForConditionalGeneration"]
            )
            self.thinker = init_vllm_registered_model(
                vllm_config=thinker_vllm_config,
                prefix=maybe_prefix(prefix, "thinker"),
                hf_config=self.thinker_config,
                architectures=["Qwen3_5OmniThinkerForConditionalGeneration"],
            )
            self.model = self.thinker
            self.talker = self.disable_talker()
            self.audio_tokenizer = self.disable_audio_tokenizer()
        elif self.model_stage == "talker":
            talker_vllm_config = vllm_config.with_hf_config(
                self.talker_config, architectures=["Qwen3_5OmniTalkerForConditionalGeneration"]
            )
            self.talker = init_vllm_registered_model(
                vllm_config=talker_vllm_config,
                prefix=maybe_prefix(prefix, "talker"),
                hf_config=self.talker_config,
                architectures=["Qwen3_5OmniTalkerForConditionalGeneration"],
            )
            self.talker.init_multi_modal(self.thinker_config)
            self.model = self.talker
            self.thinker = None
            self.audio_tokenizer = self.disable_audio_tokenizer()
            self.suppressed_tokens = self._get_talker_suppressed_tokens()
            self.requires_raw_input_tokens = True
        elif self.model_stage == "audio_tokenizer":
            audio_tokenizer_vllm_config = vllm_config.with_hf_config(
                self.audio_tokenizer_config, architectures=["Qwen3_5OmniAudioTokenizerModel"]
            )
            self.audio_tokenizer = init_vllm_registered_model(
                vllm_config=audio_tokenizer_vllm_config,
                prefix=maybe_prefix(prefix, "audio_tokenizer"),
                hf_config=self.audio_tokenizer_config,
                architectures=["Qwen3_5OmniAudioTokenizerModel"],
            )
            self.model = self.audio_tokenizer
            self.thinker = None
            self.talker = self.disable_talker()
            self.requires_raw_input_tokens = True

    def disable_talker(self):
        if hasattr(self, "talker"):
            del self.talker
        self.has_talker = False

    def disable_audio_tokenizer(self):
        if hasattr(self, "audio_tokenizer"):
            del self.audio_tokenizer
        self.has_audio_tokenizer = False

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        """Get the device of a module."""
        try:
            return next(module.parameters()).device
        except StopIteration:
            # No parameters; fall back to buffers or cpu
            for _, buf in module.named_buffers(recurse=True):
                return buf.device
            return torch.device("cpu")

    def _get_talker_suppressed_tokens(self):
        return [
            i
            for i in range(
                self.config.talker_config.text_config.vocab_size - 1024,
                self.config.talker_config.text_config.vocab_size,
            )
            if i != self.config.talker_config.codec_eos_token_id
        ]

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        if self.model_stage == "code2wav":
            return torch.zeros_like(input_ids).reshape(-1, 1).repeat(1, self.vllm_config.model_config.get_hidden_size())
        return self.model.embed_input_ids(
            input_ids=input_ids, multimodal_embeddings=multimodal_embeddings, is_multimodal=is_multimodal
        )

    def embed_multimodal(self, **kwargs):
        """Delegate to active model for multimodal processing."""
        return self.model.embed_multimodal(**kwargs)

    def _get_talker_system_parts(self, id_dtype, speaker, speaker_id, prompt_speaker_codes, system_instruct_ids):
        system_embeddings = []
        # role embedding
        role_tokens = torch.tensor(
            [[self.config.im_start_token_id, self.config.system_token_id, self.config.nl_token_id]],
            device=self.talker.device,
            dtype=id_dtype,
        )
        system_embeddings.append(self.talker.get_input_text_embeddings()(role_tokens))

        # instrurct embedding
        if system_instruct_ids is not None:
            system_embeddings.append(
                self.talker.get_input_text_embeddings()(system_instruct_ids).to(self.talker.device)
            )

        # speaker system instruct embedding
        if prompt_speaker_codes is None:
            system_embeddings.append(
                self.talker.get_input_text_embeddings()(
                    torch.tensor(
                        self.config.talker_config.speaker_system_prompt_id[speaker.lower()],
                        device=self.talker.device,
                        dtype=torch.long,
                    ).unsqueeze(0)
                ).to(self.talker.device)
            )

        # codec bos + spekear code + codec eos
        codec_bos_token = torch.tensor(
            [[self.config.talker_config.codec_bos_id]],
            device=self.talker.device,
            dtype=torch.long,
        )
        system_embeddings.append(self.talker.get_input_embeddings()(codec_bos_token).to(self.talker.device))
        if prompt_speaker_codes is not None:
            prompt_speaker_codes = prompt_speaker_codes.to(self.talker.device)
            # prompt speaker
            if prompt_speaker_codes.ndim == 2:
                prompt_speaker_codes = prompt_speaker_codes.unsqueeze(0)
            system_embeddings.append(self._get_codec_input_embeddings(prompt_speaker_codes))
        else:
            # speaker id
            speaker_code = self.talker.speaker_codec_embeddings[speaker_id].unsqueeze(0)
            system_embeddings.append(self._get_codec_input_embeddings(speaker_code))
        codec_eos_token = torch.tensor(
            [[self.config.talker_config.codec_eos_token_id]],
            device=self.talker.device,
            dtype=torch.long,
        )
        system_embeddings.append(self.talker.get_input_embeddings()(codec_eos_token).to(self.talker.device))

        # chat end embedding
        chat_end_tokens = torch.tensor(
            [[self.config.im_end_token_id, self.config.nl_token_id]],
            device=self.talker.device,
            dtype=id_dtype,
        )
        system_embeddings.append(self.talker.get_input_text_embeddings()(chat_end_tokens))
        system_embeddings = torch.cat(system_embeddings, dim=1)
        return system_embeddings

    def _get_talker_user_parts(
        self, im_start_index, segment_end_index, thinker_role_mask, thinker_hidden, thinker_to_talker_text_embeds
    ):
        user_talker_part = torch.empty(
            (1, segment_end_index - im_start_index, self.config.talker_config.text_config.hidden_size),
            device=self.talker.device,
            dtype=self.talker.dtype,
        )
        user_mask = ~thinker_role_mask[:, im_start_index:segment_end_index]
        # thinker hidden data
        if user_mask.any():
            user_thinker_hidden = thinker_hidden[:, im_start_index:segment_end_index][user_mask]
            mm_hidden = self.talker.hidden_projection(user_thinker_hidden).to(self.talker.device)
            user_talker_part[user_mask] = mm_hidden
        user_thinker_embed = thinker_to_talker_text_embeds[:, im_start_index:segment_end_index][~user_mask]
        user_talker_part[~user_mask] = user_thinker_embed
        return user_talker_part

    def _get_talker_assistant_parts(
        self, id_dtype, talker_language, talker_text_embeds, tts_bos_embed, tts_eos_embed, text_in_chunk_n
    ):
        assistant_embeddings = []

        # role embedding
        role_tokens = torch.tensor(
            [[self.config.im_start_token_id, self.config.assistant_token_id, self.config.nl_token_id]],
            device=self.talker.device,
            dtype=id_dtype,
        )
        assistant_embeddings.append(self.talker.get_input_text_embeddings()(role_tokens))

        # codec special token embedding
        if talker_language is not None and self.config.talker_language_id.get(talker_language.lower()) is not None:
            language_id = self.config.talker_language_id.get(talker_language.lower())
            codec_special_tokens = torch.tensor(
                [
                    [
                        self.config.talker_config.codec_think_id,
                        self.config.talker_config.codec_think_bos_id,
                        language_id,
                        self.config.talker_config.codec_think_eos_id,
                    ]
                ],
                device=self.talker.device,
                dtype=torch.long,
            )
        else:
            codec_special_tokens = torch.tensor(
                [
                    [
                        self.config.talker_config.codec_nothink_id,
                        self.config.talker_config.codec_think_bos_id,
                        self.config.talker_config.codec_think_eos_id,
                    ]
                ],
                device=self.talker.device,
                dtype=torch.long,
            )
        assistant_embeddings.append(self.talker.get_input_embeddings()(codec_special_tokens).to(self.talker.device))
        # assistant prefill embedding
        assistant_embeddings.append(tts_bos_embed)
        code_bos_token = torch.tensor(
            [
                [
                    self.config.talker_config.codec_bos_id,
                ]
            ],
            device=self.talker.device,
            dtype=torch.long,
        )
        assistant_embeddings.append(self.talker.get_input_embeddings()(code_bos_token).to(self.talker.device))
        # assistant text embedding
        assistant_text_embedding_list = []
        talker_text_embeds_with_eos = torch.cat([talker_text_embeds, tts_eos_embed], dim=1)
        for i in range(0, talker_text_embeds_with_eos.shape[1], text_in_chunk_n):
            assistant_text_embedding_list.append(
                talker_text_embeds_with_eos[:, i : i + text_in_chunk_n].to(self.talker.device)
            )

        assistant_embeddings.append(assistant_text_embedding_list.pop(0))
        assistant_embeddings = torch.cat(assistant_embeddings, dim=1)
        return assistant_embeddings, assistant_text_embedding_list

    def _get_codec_input_embeddings(
        self,
        codes: torch.Tensor,  # shape: batch_size, code_group_size, code_length
    ):
        codec_embeddings = torch.cat(
            [self.talker.get_input_embeddings()(codes[:, 0:1])]
            + [
                self.talker.code_predictor.get_input_embeddings()[i](codes[:, i + 1 : i + 2])
                for i in range(self.config.talker_config.num_code_groups - 1)
            ],
            dim=1,
        ).sum(1)
        return codec_embeddings

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec] | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, int]:
        if self.model_stage == "thinker":
            if mm_features is None:
                msg = "Qwen3 Omni thinker get_mrope_input_positions requires mm_features"
                raise ValueError(msg)
            return self.thinker.get_mrope_input_positions(input_tokens, mm_features)
        return MRotaryEmbedding.get_input_positions_tensor(input_tokens, **kwargs)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        generate_audio: bool = True,
        voice_type: str = "ethan",
        codec: torch.Tensor | None = None,
        sampling_metadata: SamplingMetadata | None = None,
        logits_index: int | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: object,
    ):
        if self.model_stage == "thinker":
            thinker_dev = self._module_device(self.thinker)

            # Move to thinker device
            if input_ids is not None and input_ids.device != thinker_dev:
                input_ids = input_ids.to(thinker_dev)
            if positions is not None and positions.device != thinker_dev:
                positions = positions.to(thinker_dev)
            if inputs_embeds is not None and inputs_embeds.device != thinker_dev:
                inputs_embeds = inputs_embeds.to(thinker_dev)

            # Run thinker forward
            # If talker expects a specific intermediate layer, capture it here
            accept_layer = getattr(self.talker_config, "accept_hidden_layer", None)
            capture_kwargs = {}
            if accept_layer is not None:
                capture_kwargs = {
                    "capture_layer_indices": [0, int(accept_layer)],
                    "return_hidden_states": True,
                }

            # Run thinker
            thinker_output = self.thinker(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **capture_kwargs,
                **kwargs,
            )
            if isinstance(thinker_output, tuple):
                text_hidden_states, captured_layer_dict = thinker_output
            else:
                text_hidden_states = thinker_output
                captured_layer_dict = {}
            return text_hidden_states, captured_layer_dict
        # ========== Stage 2.1: Talker ==========
        elif self.model_stage == "talker":
            if input_ids is None:
                # special case for profile run
                input_ids = torch.zeros(inputs_embeds.shape[0], dtype=torch.long, device=inputs_embeds.device)

            # Ensure we have base embeddings when only ids are provided
            if inputs_embeds is None and input_ids is not None:
                inputs_embeds = self.talker.embed_input_ids(input_ids)

            # TODO(Peiqi): temporal hack here to support voice_type.
            if not hasattr(self, "voice_type"):
                self.voice_type = voice_type

            # Run talker forward
            with torch.inference_mode():
                talker_hidden = self.talker.forward(
                    input_ids=input_ids,
                    positions=positions,
                    inputs_embeds=inputs_embeds,
                )
            return talker_hidden

        # ========== Stage 3: audio_tokenizer ==========
        elif self.model_stage == "audio_tokenizer":
            seq_token_counts: list[int] | None = kwargs.get("seq_token_counts")

            # Extract codec codes from input
            if input_ids.shape[0] % 16 == 0:
                if seq_token_counts is not None:
                    max_seq_len = max(seq_token_counts) // 16
                    batch_size = len(seq_token_counts)
                    split_codes = torch.split(input_ids, seq_token_counts, dim=0)
                    codes = torch.zeros((batch_size, 16, max_seq_len), device=input_ids.device, dtype=input_ids.dtype)
                    for idx, code in enumerate(split_codes):
                        seq_len = code.shape[0] // 16
                        codes[idx, :, :seq_len] = code.reshape(16, seq_len)
                else:
                    codes = input_ids.reshape(1, 16, -1)
            else:
                logger.warning(
                    (
                        "Input_ids length: %s is not divisible by 16, padding "
                        "with zeros. This should only happen in warm up."
                    ),
                    input_ids.shape[0],
                )
                input_ids_flatten = input_ids.reshape(-1)
                input_ids_flatten = torch.cat(
                    [
                        input_ids_flatten,
                        torch.zeros(16 - input_ids.shape[0] % 16, dtype=torch.long, device=input_ids.device),
                    ]
                )
                codes = input_ids_flatten.reshape(1, 16, -1)
            wav = self.generate_wav_from_codes(codes=codes, seq_token_counts=seq_token_counts)
            return wav

    def _get_tts_embed(self, thinker_embed, tts_bos_thinker, tts_eos_thinker, tts_pad_thinker):
        """Project thinker-side TTS embeddings into talker text space."""
        module_device = self._module_device(self.talker)

        def _ensure_1x1(x: torch.Tensor) -> torch.Tensor:
            if x.ndim == 3:
                return x[0, -1:, :]
            if x.ndim == 2:
                return x[-1]
            return x.view(1, 1, -1)

        def _proj_from_thinker(x_opt: torch.Tensor | None) -> torch.Tensor:
            if isinstance(x_opt, torch.Tensor) and x_opt.numel() > 0:
                xin = _ensure_1x1(x_opt).to(module_device)
            else:
                xin = torch.zeros(
                    (1, thinker_embed.shape[-1]),
                    device=module_device,
                    dtype=thinker_embed.dtype,
                )
            return self.talker.text_projection(xin).to(module_device)

        self.tts_bos_embed = _proj_from_thinker(tts_bos_thinker)
        self.tts_eos_embed = _proj_from_thinker(tts_eos_thinker)
        self.tts_pad_embed = _proj_from_thinker(tts_pad_thinker)
        return self.tts_bos_embed, self.tts_eos_embed, self.tts_pad_embed

    def talker_preprocess_prefill(self, input_ids: torch.Tensor, input_embeds: torch.Tensor, **info_dict: dict):
        # Containers to return per-request updates (e.g., code_predictor_hidden_per_request)
        update_dict: dict[str, dict] = {}
        # TODO(Peiqi): add voice_type support
        voice_type = self.voice_type
        start_index = info_dict.get("num_processed_tokens", 0)
        end_index = start_index + input_embeds.shape[0]
        # Read thinker outputs for prefill
        thinker_sequence_embeds = info_dict.get("thinker_embeddings").to(
            device=self._module_device(self.talker), dtype=torch.bfloat16
        )  # Tensor [P,H]
        thinker_hidden_states = info_dict.get("thinker_hidden_states").to(
            device=self._module_device(self.talker), dtype=torch.bfloat16
        )  # Tensor [K,H]
        thinker_sequences = (
            info_dict.get("thinker_sequences")
            if info_dict.get("thinker_sequences") is None
            else torch.as_tensor(info_dict.get("thinker_sequences"), device=self._module_device(self.talker))
        )
        thinker_chatml_ids = (
            info_dict.get("thinker_input_ids")
            if info_dict.get("thinker_input_ids") is None
            else torch.as_tensor(info_dict.get("thinker_input_ids"), device=self._module_device(self.talker))
        )

        tts_bos_thinker = info_dict.get("tts_bos_embed").to(
            device=self._module_device(self.talker), dtype=torch.bfloat16
        )
        tts_eos_thinker = info_dict.get("tts_eos_embed").to(
            device=self._module_device(self.talker), dtype=torch.bfloat16
        )
        tts_pad_thinker = info_dict.get("tts_pad_embed").to(
            device=self._module_device(self.talker), dtype=torch.bfloat16
        )

        if thinker_sequence_embeds is None or thinker_hidden_states is None:
            raise ValueError(
                "additional_information_by_req_id must include "
                "'thinker_embeddings' and 'thinker_hidden_states' for talker prefill."
            )

        # Normalize to tensors
        if not isinstance(thinker_sequence_embeds, torch.Tensor):
            thinker_sequence_embeds = torch.as_tensor(thinker_sequence_embeds, device=self._module_device(self.talker))
        if not isinstance(thinker_hidden_states, torch.Tensor):
            thinker_hidden_states = torch.as_tensor(thinker_hidden_states, device=self._module_device(self.talker))

        if isinstance(thinker_chatml_ids, torch.Tensor) or isinstance(thinker_chatml_ids, list):
            ids_chatml = (
                thinker_chatml_ids
                if isinstance(thinker_chatml_ids, torch.Tensor)
                else torch.as_tensor(thinker_chatml_ids, device=self._module_device(self.talker))
            )
            if ids_chatml.ndim == 1:
                ids_chatml = ids_chatml.unsqueeze(0)
        else:
            # Fallback: create dummy ids if not provided
            ids_chatml = torch.zeros(
                (1, thinker_sequence_embeds.shape[1]),
                dtype=torch.long,
                device=self._module_device(self.talker),
            )
            thinker_sequences = ids_chatml

        speaker_id = self._get_text_spk_token_id(voice_type)
        req_input_ids, req_embeds, trailing_text_hidden = self._thinker_to_talker_prefill(
            thinker_embed=thinker_sequence_embeds.to(self._module_device(self.talker)),
            thinker_hidden=thinker_hidden_states.to(self._module_device(self.talker)),
            multimodal_mask=None,
            input_ids=ids_chatml.to(self._module_device(self.talker)),
            thinker_result_ids=thinker_sequences.to(self._module_device(self.talker)),
            speaker_id=speaker_id,
            tts_bos_thinker=tts_bos_thinker,
            tts_eos_thinker=tts_eos_thinker,
            tts_pad_thinker=tts_pad_thinker,
        )

        # Queue trailing_text_hidden for decode (drop first for next steps),
        try:
            if isinstance(trailing_text_hidden, torch.Tensor) and trailing_text_hidden.numel() > 0:
                if trailing_text_hidden.ndim == 2:
                    rem_tail = trailing_text_hidden
                elif trailing_text_hidden.ndim == 1:
                    rem_tail = torch.zeros(
                        0,
                        trailing_text_hidden.shape[0],
                        dtype=trailing_text_hidden.dtype,
                        device=trailing_text_hidden.device,
                    )
                else:
                    # compatible with old shape [1,S,D]
                    rem_tail = trailing_text_hidden.squeeze(0)
                if rem_tail.shape[0] > 0:
                    update_dict["trailing_text_hidden"] = rem_tail.detach().to("cpu").contiguous()
            # Also persist projected tts_pad for decode fallback if needed
            if isinstance(tts_pad_thinker, torch.Tensor):
                pad_in = tts_pad_thinker
                if pad_in.ndim == 2:
                    pad_in = pad_in.unsqueeze(0)
                if pad_in.ndim == 1:
                    pad_in = pad_in.view(1, 1, -1)
                pad_proj = self.talker.text_projection(pad_in.to(self._module_device(self.talker)))
                update_dict["tts_pad_embed_projected"] = pad_proj.detach().to("cpu").contiguous()
        except Exception:
            pass

        return req_input_ids[start_index:end_index], req_embeds[start_index:end_index], update_dict

    def _thinker_to_talker_prefill(
        self,
        thinker_embed: torch.Tensor,
        thinker_hidden: torch.Tensor,
        multimodal_mask: torch.Tensor | None,
        input_ids: torch.Tensor,
        thinker_result_ids: torch.Tensor,
        speaker_id,
        tts_bos_thinker: torch.Tensor | None = None,
        tts_eos_thinker: torch.Tensor | None = None,
        tts_pad_thinker: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Project thinker outputs to talker inputs during prefill stage.

        Returns:
            (input_ids, input_embeds) for talker
        """
        im_start_indexes = torch.cat(
            (
                torch.nonzero(input_ids[0] == self.config.im_start_token_id).squeeze(),
                torch.tensor([thinker_result_ids.shape[-1]], device=input_ids.device, dtype=input_ids.dtype),
            ),
            dim=-1,
        )  # Shape [n_starts + 1]; Take batch 0 since batched inference is not supported here.
        multimodal_mask = (
            (thinker_result_ids == self.thinker_config.audio_token_id) |
            (thinker_result_ids == self.thinker_config.image_token_id) |
            (thinker_result_ids == self.thinker_config.video_token_id)
        ).to(input_ids.device)  # [t] # fmt: skip

        tts_bos_embed, tts_eos_embed, tts_pad_embed = self._get_tts_embed(
            thinker_embed, tts_bos_thinker, tts_eos_thinker, tts_pad_thinker
        )

        talker_input_embeds = []  # [1 t d]
        talker_input_ids = []
        trailing_text_hidden_all: torch.Tensor | None = None
        # For every chatml parts
        for i in range(len(im_start_indexes) - 1):
            im_start_index = im_start_indexes[i].item()
            segment_end_index = im_start_indexes[i + 1].item()
            role_token = input_ids[0][im_start_index + 1]
            # Talker should ignore thinker system prompt
            if (role_token == self.config.system_token_id).item():
                continue
            # Talker takes word embeddings for tokens and hidden state from `accept_hidden_layer` for multimodal inputs
            elif (role_token == self.config.user_token_id).item():
                talker_user_part = self._get_talker_user_parts(
                    im_start_index, segment_end_index, multimodal_mask, thinker_hidden, thinker_embed
                )
                talker_input_embeds.append(talker_user_part)
                talker_input_ids.append(thinker_result_ids[im_start_index:segment_end_index])
            # Take assistant output (for now)
            elif (role_token == self.config.assistant_token_id).item() and i == len(im_start_indexes) - 2:
                talker_assistant_embeds, talker_assistant_ids, trailing_text_hidden = self._get_talker_assistant_parts(
                    im_start_index,
                    segment_end_index,
                    speaker_id,
                    thinker_embed,
                    tts_pad_embed,
                    tts_bos_embed,
                    tts_eos_embed,
                )
                talker_input_embeds.append(talker_assistant_embeds)
                talker_input_ids.append(talker_assistant_ids)
                # capture trailing text hidden for decode steps
                try:
                    if isinstance(trailing_text_hidden, torch.Tensor):
                        trailing_text_hidden_all = trailing_text_hidden
                except Exception:
                    pass
            # History assistant output (ignore for now)
            elif (role_token == self.config.assistant_token_id).item() and i != len(im_start_indexes) - 2:
                continue
            else:
                raise AssertionError("Expect role id after <|im_start|> (assistant, user, system)")
        talker_input_embed = torch.cat([embed.to(input_ids.device) for embed in talker_input_embeds], dim=0)
        talker_input_id = torch.cat([embed.to(input_ids.device) for embed in talker_input_ids], dim=0)

        return talker_input_id, talker_input_embed, trailing_text_hidden_all

    def _thinker_decode_to_talker_decode(
        self,
        info_dict: dict,
        device: torch.device,
        update_dict,
    ):
        """
        Project thinker outputs to talker inputs during prefill stage.
        Returns:
            (input_ids, input_embeds) for talker
        """
        thinker_embed = info_dict.get("thinker_embeddings", None)
        start_index = info_dict.get("num_processed_tokens", 0)
        if start_index >= thinker_embed.shape[0]:
            if info_dict.get("finished_flag"):
                return self.tts_pad_embed.to(device)
            update_dict["finished_flag"] = True
            return self.tts_eos_embed.to(device)

        thinker_embed = thinker_embed[start_index : start_index + 1].to(device)
        return self.talker.text_projection(thinker_embed).to(device)

    def talker_preprocess_decode(
        self, input_ids: torch.Tensor, input_embeds: torch.Tensor, update_dict: dict, **info_dict: dict
    ):
        last_talker_hidden = None
        text_step = None
        try:
            if self.vllm_config.model_config.async_chunk:
                text_step = self._thinker_decode_to_talker_decode(info_dict, input_ids.device, update_dict)
            else:
                q_tail = info_dict.get("trailing_text_hidden", None)
                if isinstance(q_tail, torch.Tensor) and q_tail.numel() > 0:
                    use_vec = q_tail[0:1, :]
                    new_q_tail = (
                        q_tail[1:, :].detach().to("cpu").contiguous()
                        if q_tail.shape[0] > 1
                        else self.tts_pad_embed.to(input_embeds.device, dtype=input_embeds.dtype)
                    )
                    text_step = use_vec.to(input_embeds.device, dtype=input_embeds.dtype)
                    update_dict["trailing_text_hidden"] = new_q_tail
                else:
                    text_step = self.tts_pad_embed.to(input_embeds.device, dtype=input_embeds.dtype)

            last_talker_hidden_tensor = info_dict.get("last_talker_hidden")
            if last_talker_hidden_tensor is not None:
                last_talker_hidden = last_talker_hidden_tensor.to(input_embeds.device, dtype=input_embeds.dtype)
                last_talker_hidden = last_talker_hidden.reshape(*last_talker_hidden.shape[-2:])  # [1, hidden_size]
            else:
                last_talker_hidden = torch.zeros(
                    (1, self.talker_config.text_config.hidden_size),
                    device=input_embeds.device,
                    dtype=input_embeds.dtype,
                )
        except Exception as e:
            logger.error(f"Error in decode: {e}")

        return last_talker_hidden, text_step, update_dict

    def _get_talker_user_parts(self, im_start_index, segment_end_index, multimodal_mask, thinker_hidden, thinker_embed):
        user_talker_part = torch.empty(
            (segment_end_index - im_start_index, self.config.talker_config.text_config.hidden_size),
            device=thinker_hidden.device,
            dtype=torch.bfloat16,
        )

        user_mm_mask = multimodal_mask[im_start_index:segment_end_index]
        # Multimodal data exists
        if user_mm_mask.any():
            user_thinker_hidden_mm = thinker_hidden[im_start_index:segment_end_index][user_mm_mask]
            mm_hidden = self.talker.hidden_projection(user_thinker_hidden_mm).to(thinker_hidden.device)
            user_talker_part[user_mm_mask] = mm_hidden
        user_thinker_embed = thinker_embed[im_start_index:segment_end_index][~user_mm_mask]
        user_text_hidden = self.talker.text_projection(user_thinker_embed).to(thinker_hidden.device)
        user_talker_part[~user_mm_mask] = user_text_hidden
        return user_talker_part

    def _get_talker_assistant_parts(
        self, im_start_index, segment_end_index, speaker_id, thinker_embed, tts_pad_embed, tts_bos_embed, tts_eos_embed
    ):
        assistant_hidden = self.talker.text_projection(thinker_embed[im_start_index:segment_end_index]).to(
            tts_pad_embed.device
        )  # [t, d]

        # [3 tokens] + [4 pad] + [1 BOS] + [1 first text] = 9 tokens
        assistant_text_hidden = torch.cat(
            (
                assistant_hidden[:3],
                tts_pad_embed.expand(4, -1),
                tts_bos_embed,
                assistant_hidden[3:4]
                if assistant_hidden.shape[0] > 3
                else torch.zeros(
                    (1, assistant_hidden.shape[1]),
                    device=assistant_hidden.device,
                    dtype=assistant_hidden.dtype,
                ),  # First text
            ),
            dim=0,
        )
        codec_special_tokens = torch.tensor(
            [
                self.config.talker_config.codec_nothink_id,
                self.config.talker_config.codec_think_bos_id,
                self.config.talker_config.codec_think_eos_id,
                speaker_id,
                self.config.talker_config.codec_pad_id,
                self.config.talker_config.codec_bos_id,
            ],
            device=tts_pad_embed.device,
            dtype=torch.long,
        )
        embed_input_ids = self.talker.embed_input_ids(codec_special_tokens).to(
            device=tts_pad_embed.device, dtype=torch.bfloat16
        )
        assistant_codec_hidden = torch.cat(
            (
                torch.zeros(
                    (3, self.config.talker_config.text_config.hidden_size),
                    device=tts_pad_embed.device,
                    dtype=torch.bfloat16,
                ),
                embed_input_ids,
            ),
            dim=0,
        )

        if assistant_hidden.shape[0] > 4:
            trailing_text_hidden = torch.cat(
                (assistant_hidden[4:], tts_eos_embed),
                dim=0,
            )
        else:
            trailing_text_hidden = torch.zeros(
                tts_eos_embed.shape, device=tts_eos_embed.device, dtype=tts_eos_embed.dtype
            )

        input_embeds = assistant_text_hidden + assistant_codec_hidden
        input_ids = torch.full(
            (assistant_text_hidden.shape[0],),
            fill_value=self.config.tts_pad_token_id,
            dtype=torch.long,
            device=assistant_text_hidden.device,
        )
        return input_embeds, input_ids, trailing_text_hidden

    @torch.inference_mode()
    def extract_prompt_speaker_codes(self, prompt_wav: torch.Tensor) -> torch.Tensor:
        prompt_wav = prompt_wav.reshape(1, -1)
        prompt_speaker_codes = self.audio_tokenizer.encode(prompt_wav).audio_codes
        return prompt_speaker_codes

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs) -> OmniOutput:
        """
        Make an OmniOutput object from model outputs.
        Args:
            model_outputs: Model outputs
        """
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        if self.model_stage == "thinker":
            text_hidden_states, captured_layer_dict = model_outputs
            # Compute thinker-side TTS token embeddings for BOS/EOS/PAD and expose via multimodal outputs.
            # These will later be projected into talker text space by the talker stage.
            multimodal_outputs = captured_layer_dict if captured_layer_dict is not None else {}
            try:
                thinker_tts_embeds = self.thinker.embed_input_ids(self.tts_tokens)  # [1,3,thinker_hidden]
                if (
                    isinstance(thinker_tts_embeds, torch.Tensor)
                    and thinker_tts_embeds.ndim == 3
                    and thinker_tts_embeds.shape[1] == 3
                ):
                    bos_eos_pad = thinker_tts_embeds.to(text_hidden_states.device).chunk(3, dim=1)  # 3 * [1,1,H]
                    multimodal_outputs["tts_bos_embed"] = [bos_eos_pad[0]]
                    multimodal_outputs["tts_eos_embed"] = [bos_eos_pad[1]]
                    multimodal_outputs["tts_pad_embed"] = [bos_eos_pad[2]]
            except Exception:
                # Best-effort; absence will be handled by talker with fallbacks
                pass

            # Return text-only output (with multimodal sidecar)
            return OmniOutput(
                text_hidden_states=(text_hidden_states.reshape(-1, text_hidden_states.shape[-1])),
                multimodal_outputs=multimodal_outputs,
            )
        elif self.model_stage == "talker":
            talker_hidden = model_outputs
            # merge the code_predictor_codes from the info_dict list into a single tensor
            multimodal_outputs: dict = None
            # Here is the only place to use model_intermediate_buffer. After MTP in the
            # preprocess function, the code_predictor_codes are stored in the info_dict list.
            # We need to merge the tensors from different requests into a single tensor.
            # In the future, we may allow user to custom an aggregated function.
            info_dicts = kwargs.get("model_intermediate_buffer")
            if info_dicts is None:
                info_dicts = kwargs.get("runtime_additional_information")

            if "runtime_additional_information" in kwargs and "model_intermediate_buffer" not in kwargs:
                logger.warning_once("runtime_additional_information is deprecated, use model_intermediate_buffer")
            code_predictor_codes = [info.get("code_predictor_codes") for info in info_dicts]
            multimodal_outputs = {"code_predictor_codes": torch.cat(code_predictor_codes, dim=0)}
            span_len = multimodal_outputs["code_predictor_codes"].shape[0]
            talker_hidden = talker_hidden[:span_len]
            return OmniOutput(text_hidden_states=talker_hidden, multimodal_outputs=multimodal_outputs)
        elif self.model_stage == "audio_tokenizer":
            audio_tensors = model_outputs
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [audio_tensor.reshape(1, -1) for audio_tensor in audio_tensors]},
            )

        return model_outputs

    @torch.inference_mode()
    def generate_wav_from_codes(
        self, codes: torch.Tensor, chunk_size: int = 300, left_context_size: int = 25,
            seq_token_counts: list[int] | None = None,
    ) -> list[torch.Tensor]:
        batch_size = codes.shape[0]
        per_request_wavs: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
        start_index = 0
        while start_index < codes.shape[-1]:
            end_index = min(start_index + chunk_size, codes.shape[-1])
            context_size = left_context_size if start_index - left_context_size > 0 else start_index
            codes_chunk = codes[..., start_index - context_size : end_index]
            audio_values = self.audio_tokenizer.decode(codes_chunk.transpose(-1, -2)).audio_values
            for i, wav in enumerate(audio_values):
                per_request_wavs[i].append(
                    wav[..., context_size * self.audio_tokenizer.decode_upsample_rate:]
                )
            start_index = end_index

        if seq_token_counts is not None:
            code_seq_lens = [seq_len // self.config.num_quantizers for seq_len in seq_token_counts]
        else:
            code_seq_lens = [codes.shape[-1]] * batch_size
        result = []
        for i in range(batch_size):
            full_wav = torch.cat(per_request_wavs[i], dim=-1)
            result.append(full_wav[: code_seq_lens[i] * self.audio_tokenizer.decode_upsample_rate])
        return result

    def _warn_talker_sampling_temperature(self, sampling_metadata: SamplingMetadata):
        warning_parts = []
        if sampling_metadata.temperature is None:
            warning_parts.append(
                "Temperature is set to None, as all requests are greedy. "
                "This is equivalent to setting temperature to 0.0."
                "Please consider setting a higher temperature i.e. 0.4."
            )
        else:
            warning_parts.append(
                "Temperature is set to: "
                f"{sampling_metadata.temperature}, where temperature as 0.0 may "
                "cause repetitive output. Please consider setting a higher "
                "temperature i.e. 0.4."
            )
        warning_parts.append(
            "This warning will be shown only once, for the first request where "
            "temperature is 0.0. Later requests will not show this warning but "
            "still be affected by the temperature."
        )
        warning_info = "\n".join(warning_parts)
        logger.warning_once(warning_info)

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: SamplingMetadata = None,
    ) -> torch.Tensor | None:
        """Compute logits from hidden states."""
        # Handle OmniOutput type
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states

        if (
            getattr(self, "model_stage", None) == "talker"
            and sampling_metadata is not None
            and (sampling_metadata.temperature is None or (sampling_metadata.temperature <= 0).any())
        ):
            self._warn_talker_sampling_temperature(sampling_metadata)

        # Use active model for logits computation
        logits = self.model.compute_logits(hidden_states)  # V, d
        # Talker: suppress tokens by setting their probability to ~1e-9 (finite very small),
        # implemented by assigning their logits to log(1e-9).

        if getattr(self, "model_stage", None) == "talker" and isinstance(logits, torch.Tensor):
            try:
                logits_cpu = logits.cpu()
                logits_cpu[:, self.suppressed_tokens] = -1e9
                logits = logits_cpu.to(logits.device)
            except Exception as e:
                print(f"Error in logits suppression: {e}")
                print(f"logits.shape: {logits.shape}")
                print(f"self.suppressed_tokens: {self.suppressed_tokens}")
                raise e
            logits[:, self.suppressed_tokens] = -1e9
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        """Sample from logits."""
        return self.model.sample(logits, sampling_metadata)

    # ==================== Weight Loading ====================

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights for all components of the omni model."""
        loaded_weights = set()
        thinker_weights = []
        talker_weights = []
        audio_tokenizer_weights = []

        # Separate weights by component
        for k, v in weights:
            if k.startswith("thinker."):
                thinker_weights.append((k, v))
            elif k.startswith("talker."):
                talker_weights.append((k, v))
            elif k.startswith("audio_tokenizer."):
                audio_tokenizer_weights.append((k, v))
            else:
                logger.warning(f"Unknown weight prefix: {k}")
        # Load thinker weights
        if self.thinker and thinker_weights:
            thinker_loaded = self.thinker.load_weights(thinker_weights)
            thinker_loaded = add_prefix_to_loaded_weights(thinker_loaded, "thinker")
            loaded_weights.update(thinker_loaded)

        # Load talker weights
        if self.talker and talker_weights:
            talker_loaded = self.talker.load_weights(talker_weights)
            talker_loaded = add_prefix_to_loaded_weights(talker_loaded, "talker")
            loaded_weights.update(talker_loaded)

        # Load audio_tokenizer weights
        if self.audio_tokenizer and audio_tokenizer_weights:
            audio_tokenizer_loaded = self.audio_tokenizer.load_weights(audio_tokenizer_weights)
            audio_tokenizer_loaded = add_prefix_to_loaded_weights(audio_tokenizer_loaded, "audio_tokenizer")
            loaded_weights.update(audio_tokenizer_loaded)

        # Log summary
        logger.info(
            "Loaded %d weights for Qwen3OmniMoe (stage=%s)",
            len(loaded_weights),
            self.model_stage,
        )

        return loaded_weights

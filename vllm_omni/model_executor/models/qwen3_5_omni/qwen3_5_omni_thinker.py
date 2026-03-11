# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3.5-Omni model (thinker part)."""

import math
from collections.abc import Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BatchFeature
from transformers.models.whisper import WhisperFeatureExtractor
from transformers.video_utils import VideoMetadata
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    IsHybrid,
    MultiModalEmbeddings,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniAudioFeatureInputs,
    merge_interleaved_embeddings,
)
from vllm.model_executor.models.qwen2_audio import (
    Qwen2AudioFeatureInputs,
    Qwen2AudioProcessingInfo,
)
from vllm.model_executor.models.qwen2_vl import (
    _create_qwen2vl_field_factory,
)
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5_MoeMixtureOfExperts,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
)
from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeAudioEncoder,
    Qwen3OmniMoeThinkerDummyInputsBuilder,
)

# yapf: enable
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessingInfo,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    _merge_multimodal_embeddings,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor,
    MultiModalPromptUpdates,
    PlaceholderFeaturesInfo,
    PromptReplacement,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors

from vllm_omni.transformers_utils.configs.configuration_qwen3_5_omni import (
    Qwen3_5OmniConfig,
    Qwen3_5OmniThinkerConfig,
)
from vllm_omni.transformers_utils.configs.processing_qwen3_5_omni import (
    Qwen3_5OmniProcessor,
)

try:
    import flash_attn
except (ImportError, ModuleNotFoundError):
    flash_attn = None

logger = init_logger(__name__)


def _get_feat_extract_output_lengths(input_lengths: torch.Tensor, downsample_times=4, chunk_size=100) -> torch.Tensor:
    input_lengths_leave = input_lengths % chunk_size
    for _ in range(downsample_times):
        input_lengths_leave = (input_lengths_leave - 1) // 2 + 1
    output_lengths = input_lengths_leave + (input_lengths // chunk_size) * math.ceil(100 / 2**downsample_times)
    return output_lengths


def check_interleaved_audio_video(
    is_video: torch.Tensor,
    is_audio: torch.Tensor,
    num_video: int,
    num_audio: int,
) -> bool:
    """
    Check if video and audio positions are interleaved in the multimodal region.

    Returns:
        True if video and audio tokens are interleaved, False otherwise.
    """
    if num_video == 0 or num_audio == 0:
        return False

    video_pos = is_video.nonzero(as_tuple=True)[0]
    audio_pos = is_audio.nonzero(as_tuple=True)[0]

    # Quick range-overlap pre-check (necessary but not sufficient).
    if not (video_pos[0].item() < audio_pos[-1].item() and audio_pos[0].item() < video_pos[-1].item()):
        return False

    # Qwen3.5-omni has timestamps between audio_in_video segments like:
    # <|vision_start|><|video_pad|><|video_pad|><|vision_end|><|audio_pad|>
    # <|audio_pad|><5.0 seconds><|vision_start|><|video_pad|>...
    # which causes Qwen2.5-Omni's density check no longer work.
    audio_diff = (
        torch.cat(
            [
                torch.tensor([False], device=is_audio.device),
                is_audio,
                torch.tensor([False], device=is_audio.device),
            ]
        )
        .int()
        .diff()
    )
    video_diff = (
        torch.cat(
            [
                torch.tensor([False], device=is_video.device),
                is_video,
                torch.tensor([False], device=is_video.device),
            ]
        )
        .int()
        .diff()
    )
    audio_starts = (audio_diff == 1).nonzero(as_tuple=True)[0].tolist()
    video_ends = (video_diff == -1).nonzero(as_tuple=True)[0].tolist()
    # We assume all video chunk have audio nested right after it with audio in video.
    # One <|vision_end|> sep token between video and audio chunk.
    return all((start - end) == 1 for start, end in zip(audio_starts, video_ends))


class Qwen3_5OmniThinkerProcessingInfo(Qwen2AudioProcessingInfo, Qwen3VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3_5OmniConfig).thinker_config

    def get_hf_processor(self, **kwargs: object) -> Qwen3_5OmniProcessor:
        processor = self.ctx.get_hf_processor(
            Qwen3_5OmniProcessor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )
        if not hasattr(processor, "audio_token"):
            processor.audio_token = "<|audio_pad|>"
        if not hasattr(processor, "image_token"):
            processor.image_token = "<|image_pad|>"
        if not hasattr(processor, "video_token"):
            processor.video_token = "<|video_pad|>"
        return processor

    def get_feature_extractor(self, **kwargs: object):
        hf_processor = self.get_hf_processor(**kwargs)
        feature_extractor = hf_processor.feature_extractor  # type: ignore
        assert isinstance(feature_extractor, WhisperFeatureExtractor)
        return feature_extractor

    def get_data_parser(self):
        feature_extractor = self.get_feature_extractor()
        return MultiModalDataParser(
            target_sr=feature_extractor.sampling_rate,
            video_needs_metadata=True,
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": None, "image": None, "video": None}


class Qwen3_5OmniThinkerDummyInputsBuilder(Qwen3VLDummyInputsBuilder):
    def get_dummy_text(self, mm_counts):
        dummy_text = super().get_dummy_text(mm_counts)
        num_audios = mm_counts.get("audio", 0)
        audio_token = "<|audio_start|><|audio_pad|><|audio_end|>"
        return audio_token * num_audios + dummy_text

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        dummy_mm_data = super().get_dummy_mm_data(seq_len, mm_counts, mm_options)
        num_audios = mm_counts.get("audio", 0)
        audio_overrides = mm_options.get("audio") if mm_options else None

        feature_extractor = self.info.get_feature_extractor()

        target_audio_length = (
            min(
                feature_extractor.chunk_length,
                30,
            )
            * feature_extractor.sampling_rate
        )

        return {
            "audio": Qwen3OmniMoeThinkerDummyInputsBuilder._get_dummy_audios(
                self,
                length=target_audio_length,
                num_audios=num_audios,
                overrides=audio_overrides,
            ),
        } | dummy_mm_data


class Qwen3_5OmniThinkerMultiModalProcessor(BaseMultiModalProcessor[Qwen3_5OmniThinkerProcessingInfo]):
    def _derive_audio_from_video_placeholders(
        self,
        placeholders: Mapping[str, list[PlaceholderFeaturesInfo]],
        mm_prompt_updates: MultiModalPromptUpdates,
        audio_modality_map: dict,
        use_audio_in_video_list: list[bool] | None = None,
    ) -> Mapping[str, list[PlaceholderFeaturesInfo]]:
        """
        Helper to derive audio placeholders from video placeholders when
        use_audio_in_video=True.
        """
        if "video" not in placeholders:
            return placeholders

        # Validate audio and video counts match
        num_videos = len(placeholders["video"])
        num_audios = len(mm_prompt_updates.get("audio", []))
        if num_audios != num_videos:
            raise ValueError(
                f"use_audio_in_video requires equal number of audio and video items, got {num_audios=}, {num_videos=}"
            )

        tokenizer = self.info.get_tokenizer()
        processor = self.info.get_hf_processor()
        audio_token_id = tokenizer.get_vocab()[processor.audio_token]

        result_placeholders = dict(placeholders)

        if use_audio_in_video_list is None:
            use_audio_in_video_list = [True] * len(placeholders["video"])
        # Each video is paired with one audio
        for video_idx, (video_placeholder, use_audio_in_video) in enumerate(
            zip(placeholders["video"], use_audio_in_video_list)
        ):
            if not use_audio_in_video:
                continue
            # Create is_embed mask selecting only audio tokens
            audio_is_embed = torch.tensor(video_placeholder.tokens) == audio_token_id

            audio_placeholder = PlaceholderFeaturesInfo(
                modality="audio",
                item_idx=audio_modality_map["video"][video_idx],
                start_idx=video_placeholder.start_idx,
                tokens=video_placeholder.tokens,
                is_embed=audio_is_embed,
            )
            result_placeholders["audio"].append(audio_placeholder)

        return result_placeholders

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        fields_config = _create_qwen2vl_field_factory(self.info.get_hf_config().vision_config.spatial_merge_size)(
            hf_inputs
        )
        audio_feature_lengths = hf_inputs.get("feature_attention_mask", torch.empty((0, 0))).sum(-1)

        return (
            dict(
                input_audio_features=MultiModalFieldConfig.flat_from_sizes("audio", audio_feature_lengths, dim=1),
                feature_attention_mask=MultiModalFieldConfig.batched("audio"),
                # audio_feature_lengths=MultiModalFieldConfig.batched("audio"),
                use_audio_in_video=MultiModalFieldConfig.batched("video"),
            )
            | fields_config
        )

    def _find_audio_modality_for_audio_in_video(self, prompt_ids: list[int]) -> dict:
        """
        Find the index of the audio token in the prompt.
        """
        video_token_id = self.info.get_hf_config().video_token_id
        audio_token_id = self.info.get_hf_config().audio_token_id
        audio_modality = {"video": [], "audio": []}
        audio_idx = 0
        for id in prompt_ids:
            if id == video_token_id:
                audio_modality["video"].append(audio_idx)
                audio_idx += 1
            elif id == audio_token_id:
                audio_modality["audio"].append(audio_idx)
                audio_idx += 1
        return audio_modality

    def _maybe_apply_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        prompt_ids: list[int],
        mm_kwargs: MultiModalKwargsItems,
        mm_prompt_updates: MultiModalPromptUpdates,
        is_update_applied: bool,
    ) -> tuple[list[int], str, Mapping[str, list[PlaceholderFeaturesInfo]]]:
        """
        Qwen3-Omni reimplements this function to handle `use_audio_in_video`.
        """
        mm_item_counts = mm_items.get_all_counts()
        self._validate_mm_kwargs(mm_kwargs, mm_item_counts)

        use_audio_in_video = []
        if "video" in mm_kwargs:
            for item in mm_kwargs["video"]:
                if item and item["use_audio_in_video"].data:
                    use_audio_in_video.append(True)
                else:
                    use_audio_in_video.append(False)
            # for mutilmodality cache
            # TODO: each item has different use audio in video
            if any([item is None for item in mm_kwargs["video"]]):
                video_token_id = self.info.get_hf_config().video_token_id
                audio_token_id = self.info.get_hf_config().audio_token_id
                video_audio_item_num = sum(id in (video_token_id, audio_token_id) for id in prompt_ids)
                if video_audio_item_num != len(mm_prompt_updates.get("video", [])) + len(
                    mm_prompt_updates.get("audio", [])
                ):
                    use_audio_in_video = len(use_audio_in_video) * [True]

        if is_update_applied:
            mm_placeholders = self._find_mm_placeholders(
                prompt_ids,
                mm_prompt_updates,
            )
            self._validate_mm_placeholders(
                mm_placeholders,
                mm_item_counts,
            )
        else:
            if any(use_audio_in_video) and "audio" in mm_prompt_updates:
                audio_modality_map = self._find_audio_modality_for_audio_in_video(prompt_ids)
                # TODO: each item has different use audio in video
                filtered_updates = {k: v for k, v in mm_prompt_updates.items() if k != "audio"}
                if len(audio_modality_map["audio"]) > 0:
                    filtered_updates["audio"] = [mm_prompt_updates["audio"][i] for i in audio_modality_map["audio"]]
                prompt_ids, mm_placeholders = self._apply_prompt_updates(
                    prompt_ids,
                    filtered_updates,
                )
                if "audio" in mm_placeholders:
                    for placeholder_index, raw_index in enumerate(audio_modality_map["audio"]):
                        mm_placeholders["audio"][placeholder_index].item_idx = raw_index
                else:
                    mm_placeholders["audio"] = []
                # Derive audio placeholders from video placeholders
                filtered_updates = {k: v for k, v in mm_prompt_updates.items() if k != "audio"}
                filtered_updates["audio"] = [mm_prompt_updates["audio"][i] for i in audio_modality_map["video"]]
                mm_placeholders = self._derive_audio_from_video_placeholders(
                    mm_placeholders,
                    filtered_updates,
                    audio_modality_map,
                    use_audio_in_video,
                )
                mm_placeholders["audio"] = sorted(mm_placeholders["audio"], key=lambda x: x.item_idx)
            else:
                prompt_ids, mm_placeholders = self._apply_prompt_updates(
                    prompt_ids,
                    mm_prompt_updates,
                )
            self._validate_mm_placeholders(
                mm_placeholders,
                mm_item_counts,
            )

        return prompt_ids, mm_placeholders

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        feature_extractor = self.info.get_feature_extractor()
        image_processor = self.info.get_image_processor(**hf_processor_mm_kwargs)
        hf_config = self.info.get_hf_config()

        audio_token = hf_processor.audio_token
        video_token = hf_processor.video_token
        image_token = hf_processor.image_token

        video_token_id = hf_config.video_token_id
        vision_start_token_id = hf_config.vision_start_token_id
        vision_end_token_id = hf_config.vision_end_token_id

        audio_token_id = hf_config.audio_token_id

        image_token_id = hf_config.image_token_id

        merge_length = image_processor.merge_size**2

        downsample_times = hf_processor_mm_kwargs.get("downsample_times", 4)
        audio_tokens_per_second = math.ceil(
            feature_extractor.sampling_rate / feature_extractor.hop_length / 2**downsample_times
        )

        def get_audio_replacement_qwen35omni(item_idx: int):
            out_item = out_mm_kwargs["audio"][item_idx]

            num_features = _get_feat_extract_output_lengths(out_item["feature_attention_mask"].data.sum())

            timestamp_interval = hf_processor_mm_kwargs.get("timestamp_interval", 60)

            audio_token_str = hf_processor._get_audio_tokens(
                num_features,
                audio_tokens_per_second,
                timestamp_interval,
            ).replace("<|audio_placeholder|>", audio_token)
            return PromptUpdateDetails.select_token_id(audio_token_str, audio_token_id)

        def get_image_replacement_qwen35omni(item_idx: int):
            out_item = out_mm_kwargs["image"][item_idx]
            grid_thw = out_item["image_grid_thw"].data
            assert isinstance(grid_thw, torch.Tensor)

            num_tokens = int(grid_thw.prod()) // merge_length
            return [image_token_id] * num_tokens

        def get_video_replacement_qwen35omni(item_idx: int):
            video, metadata = mm_items["video"][item_idx]
            out_video_item = out_mm_kwargs["video"][item_idx]
            video_grid_thw = out_video_item["video_grid_thw"].data
            use_audio = out_video_item["use_audio_in_video"].data

            if use_audio:
                out_audio_item = out_mm_kwargs["audio"][item_idx]
                audio_num_features = _get_feat_extract_output_lengths(
                    out_audio_item["feature_attention_mask"].data.sum()
                )

            video_token_str = (
                hf_processor._get_video_tokens(
                    metadata["frames_indices"],
                    metadata["fps"],
                    video_grid_thw,
                    image_processor.merge_size,
                    audio_tokens_per_second if use_audio else None,
                    audio_num_features if use_audio else None,
                )
                .replace("<|audio_placeholder|>", audio_token)
                .replace("<|video_placeholder|>", video_token)
            )
            return PromptUpdateDetails.select_token_id(video_token_str, video_token_id)

        return [
            PromptReplacement(
                modality="audio",
                target=[audio_token_id],
                replacement=get_audio_replacement_qwen35omni,
            ),
            PromptReplacement(
                modality="image",
                target=image_token,
                replacement=get_image_replacement_qwen35omni,
            ),
            PromptReplacement(
                modality="video",
                target=[vision_start_token_id, video_token_id, vision_end_token_id],
                replacement=get_video_replacement_qwen35omni,
            ),
        ]

    def _call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        audios = mm_data.pop("audios", [])

        # NOTE: WhisperFeatureExtractor cannot handle empty list of audios
        if audios:
            # NOTE: Qwen2.5-Omni processor accept "audio"
            mm_data["audio"] = audios
        num_videos = len(mm_data.get("videos", []))
        # TODO: each item has different use audio in video
        if mm_kwargs.get("use_audio_in_video", True):
            prompt = prompt.replace("<|audio_start|><|audio_pad|><|audio_end|>", "", num_videos)

        mm_data = dict(mm_data)
        video_mm_data = {}
        video_mm_kwargs = {}
        if "videos" in mm_data and isinstance(mm_data["videos"], list) and len(mm_data["videos"]) > 0:
            video_items = mm_data.pop("videos", [])
            video_mm_data = {"videos": [], "video_metadata": []}
            for video_array, metadata in video_items:
                video_mm_data["videos"].append(video_array)
                _metadata = VideoMetadata(**{k: metadata[k] for k in metadata if k != "do_sample_frames"})
                video_mm_data["video_metadata"].append(_metadata)
            if "do_sample_frames" not in mm_kwargs:
                # qwen_vl_utils already has "do_sample_frames" in
                # mm_kwargs, don't overwrite it.
                video_mm_kwargs["do_sample_frames"] = metadata.get("do_sample_frames", False)
        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data | video_mm_data,
            mm_kwargs=mm_kwargs | video_mm_kwargs,
            tok_kwargs=tok_kwargs,
        )
        hf_inputs = BatchFeature(processed_outputs)

        input_features = hf_inputs.pop("input_features", None)
        feature_attention_mask = hf_inputs.get("feature_attention_mask", None)
        if "input_audio_features" not in hf_inputs and input_features is not None:
            if feature_attention_mask is not None:
                input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
            hf_inputs["input_audio_features"] = input_features
            hf_inputs["audio_feature_lengths"] = feature_attention_mask.sum(dim=1)

        return hf_inputs


class Qwen3_5OmniAudioEncoder(Qwen3OmniMoeAudioEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.conv2d4 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv_out = nn.Linear(
            config.downsample_hidden_size * (((((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2 + 1) // 2),
            config.d_model,
            bias=False,
        )

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor,
        aftercnn_lens: torch.Tensor,
    ):
        r"""
        feature_lens (`torch.LongTensor` of shape `(batch_size,)`):
            mel length
        aftercnn_lens (`torch.LongTensor` of shape `(batch_size,)`):
            mel length after cnn
        """
        aftercnn_lens = _get_feat_extract_output_lengths(feature_lens).long()
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum().item(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = (feature_lens % (self.n_window * 2)).long()
        chunk_lengths[chunk_lengths == 0] = self.n_window * 2

        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
        feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
        padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
            [torch.ones(length, dtype=torch.bool, device=padded_feature.device) for length in feature_lens_after_cnn],
            batch_first=True,
        )
        padded_feature = padded_feature.unsqueeze(1)
        # Split to chunk to avoid OOM during convolution
        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            padded_embed = F.gelu(self.conv2d1(chunk))
            padded_embed = F.gelu(self.conv2d2(padded_embed))
            padded_embed = F.gelu(self.conv2d3(padded_embed))
            padded_embed = F.gelu(self.conv2d4(padded_embed))
            padded_embeds.append(padded_embed)
        padded_embed = torch.cat(padded_embeds, dim=0)
        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

        positional_embedding = (
            self.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding
        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_chunk_lens = [0]
        window_aftercnn = int(padded_mask_after_cnn.shape[-1] * (self.n_window_infer // (self.n_window * 2)))
        for cnn_len in aftercnn_lens.tolist():
            cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
            remainder = cnn_len % window_aftercnn
            if remainder != 0:
                cu_chunk_lens += [remainder]
        cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(-1, dtype=torch.int32)
        max_seqlen = self.compute_attn_mask_seqlen(cu_seqlens)

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                cu_seqlens,
                max_seqlen,
            )

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return hidden_states


class Qwen3_5OmniConditionalGenerationMixin:
    _parse_and_validate_image_input = Qwen3VLForConditionalGeneration._parse_and_validate_image_input
    _parse_and_validate_video_input = Qwen3VLForConditionalGeneration._parse_and_validate_video_input
    _process_video_input = Qwen3VLForConditionalGeneration._process_video_input
    _process_image_input = Qwen3VLForConditionalGeneration._process_image_input

    def _parse_and_validate_audio_input(self, **kwargs: object) -> Qwen2_5OmniAudioFeatureInputs | None:
        input_audio_features = kwargs.pop("input_audio_features", None)
        audio_feature_lengths = kwargs.pop("audio_feature_lengths", None)
        feature_attention_mask = kwargs.pop("feature_attention_mask", None)
        if input_audio_features is None:
            return None

        return Qwen2_5OmniAudioFeatureInputs(
            type="audio_features",
            input_features=input_audio_features,
            audio_feature_lengths=audio_feature_lengths,
            feature_attention_mask=feature_attention_mask,
            validate=False,
        )

    def _process_audio_input(
        self,
        audio_input: Qwen2AudioFeatureInputs,
        audio_hashes: list[str] = None,
        cached_audio_features: torch.Tensor = None,
    ) -> torch.Tensor:
        input_features = audio_input["input_features"]
        feature_attention_mask = audio_input["feature_attention_mask"]
        if isinstance(feature_attention_mask, list):
            feature_lengths = torch.tensor(
                [mask.sum(-1) for mask in feature_attention_mask],
                dtype=torch.long,
                device=input_features.device,
            )
        else:
            feature_lengths = feature_attention_mask.sum(-1)

        if input_features.ndim == 3:
            assert input_features.shape[0] == 1
            input_features = input_features.squeeze(0)

        audio_output_lengths = _get_feat_extract_output_lengths(feature_lengths)

        audio_outputs = self.audio_tower(
            input_features.to(self.audio_tower.dtype),
            feature_lens=feature_lengths,
            aftercnn_lens=audio_output_lengths,
        )
        return audio_outputs.split(audio_output_lengths.long().tolist())


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3_5OmniThinkerMultiModalProcessor,
    info=Qwen3_5OmniThinkerProcessingInfo,
    dummy_inputs=Qwen3_5OmniThinkerDummyInputsBuilder,
)
class Qwen3_5OmniThinkerForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsMRoPE,
    Qwen3_5OmniConditionalGenerationMixin,
    IsHybrid,
    Qwen3_5_MoeMixtureOfExperts,
):
    _iter_mm_grid_hw = Qwen3VLForConditionalGeneration._iter_mm_grid_hw
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "thinker.lm_head.": "language_model.lm_head.",
            "thinker.model.": "language_model.model.",
            "thinker.": "",
        }
    )

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"
        if modality.startswith("audio"):
            return "<|audio_start|><|audio_pad|><|audio_end|>"

        raise ValueError("Only image, video or audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        thinker_config: Qwen3_5OmniThinkerConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = thinker_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"

        # force "use_flash_attention_2=True" to audio tower to align
        # the results.
        if flash_attn is not None:
            audio_config = thinker_config.audio_config
            audio_config._attn_implementation_autoset = True
            audio_config._attn_implementation = "flash_attention_2"
        else:
            logger.warning(
                "flash_attn is not available, the model may not yield the "
                "exactly same result as the transformers implementation "
                "in the audio tower part."
            )

        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_tower = Qwen3_5OmniAudioEncoder(thinker_config.audio_config)

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                vision_config=thinker_config.vision_config,
                norm_eps=getattr(thinker_config.text_config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )
        self.quant_config = quant_config

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_5MoeForCausalLM(
                vllm_config=vllm_config.with_hf_config(
                    thinker_config.text_config,
                ),
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

        # set MoE hyperparameters
        self.set_moe_parameters()

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if input_key in ("pixel_values", "image_embeds") and "image" not in mm_input_by_modality:
                mm_input_by_modality["image"] = self._parse_and_validate_image_input(**kwargs)
            if input_key in ("pixel_values_videos", "video_embeds") and "video" not in mm_input_by_modality:
                mm_input_by_modality["video"] = self._parse_and_validate_video_input(**kwargs)
            if input_key in ("input_audio_features") and "audio" not in mm_input_by_modality:
                mm_input_by_modality["audio"] = self._parse_and_validate_audio_input(**kwargs)
        return mm_input_by_modality

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return []

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                vision_embeddings = self._process_image_input(multimodal_input)
                multimodal_embeddings += vision_embeddings
            if modality == "video":
                video_embeddings = self._process_video_input(multimodal_input)
                multimodal_embeddings += video_embeddings
            if modality == "audio":
                audio_embeddings = self._process_audio_input(multimodal_input)
                multimodal_embeddings += audio_embeddings
        return multimodal_embeddings

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        video_token_id = self.config.video_token_id
        audio_token_id = self.config.audio_token_id
        image_token_id = self.config.image_token_id
        is_multimodal = (input_ids == video_token_id) | (input_ids == audio_token_id) | (input_ids == image_token_id)

        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=False,  # TODO Compatible with v0.17.0
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        # Detect interleaved audio-in-video early, since it affects
        # the final embedding merge.
        is_video = is_multimodal & (input_ids == video_token_id)
        is_audio = is_multimodal & (input_ids == audio_token_id)
        num_video = is_video.sum().item()
        num_audio = is_audio.sum().item()

        is_interleaved = check_interleaved_audio_video(is_video, is_audio, num_video, num_audio)

        if is_interleaved:
            return merge_interleaved_embeddings(
                inputs_embeds,
                multimodal_embeddings,
                is_video,
                is_audio,
                is_multimodal,
                num_video,
                num_audio,
            )

        # Default: standard merge (no interleaving)
        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["talker.", "audio_tokenizer.", "mtp."],
        )
        loaded_weights = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

        return loaded_weights

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        kwargs = MultiModalFeatureSpec.gather_kwargs(
            mm_features,
            {
                "image_grid_thw",
                "video_grid_thw",
                "context_len",
                "seq_len",
            },
        )
        image_grid_thw = kwargs.get("image_grid_thw", [])
        video_grid_thw = kwargs.get("video_grid_thw", [])
        context_len = kwargs.get("context_len", 0)
        seq_len = kwargs.get("seq_len", None)

        config = self.config
        video_grid_thw = [[1, h, w] for t, h, w in video_grid_thw for _ in range(t)]

        image_token_id = config.image_token_id
        video_token_id = config.video_token_id
        audio_token_id = config.audio_token_id
        vision_start_token_id = config.vision_start_token_id
        video_end_token_id = config.vision_end_token_id
        spatial_merge_size = config.vision_config.spatial_merge_size

        input_tokens_tensor = torch.tensor(input_tokens)
        vision_start_indices = torch.argwhere(input_tokens_tensor == vision_start_token_id).squeeze(1)
        vision_tokens = input_tokens_tensor[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        llm_pos_ids_list: list = []

        st = 0
        remain_images, remain_videos = image_nums, video_nums

        image_index, video_index = 0, 0
        for _ in range(image_nums + video_nums):
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t, llm_grid_h, llm_grid_w = (
                t,
                h // spatial_merge_size,
                w // spatial_merge_size,
            )
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            vision_pos_ids = torch.stack([t_index, h_index, w_index]) + text_len + st_idx
            # explicitly add vision end token id for possible audio part
            vision_pos_ids = torch.cat(
                [vision_pos_ids, torch.full((3, 1), vision_pos_ids.max() + 1)],
                dim=-1,
            )
            # video with audio
            # since we use `llm_pos_ids_list[-1].max()` to compute text position ids
            # and audio pos ids could be smaller than vision's, we should cat audio
            # part after vision part
            possible_audio_token_idx = input_tokens.index(video_end_token_id, st) + 1
            audio_length = 0
            for token in input_tokens[possible_audio_token_idx:]:
                if token == audio_token_id:
                    audio_length += 1
                else:
                    break
            if audio_length > 0:
                audio_pos_ids = torch.arange(audio_length).view(1, -1).expand(3, -1) + text_len + st_idx
                vision_pos_ids = torch.cat([vision_pos_ids, audio_pos_ids], dim=-1)
            llm_pos_ids_list.append(vision_pos_ids)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w + 1 + audio_length

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens)).item()
        llm_positions = llm_positions[:, context_len:seq_len]
        return llm_positions, mrope_position_delta

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig) -> tuple[torch.dtype, torch.dtype]:
        return Qwen3_5MoeForConditionalGeneration.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig) -> tuple[tuple[int, int], tuple[int, int, int]]:
        return Qwen3_5MoeForConditionalGeneration.get_mamba_state_shape_from_config(vllm_config)

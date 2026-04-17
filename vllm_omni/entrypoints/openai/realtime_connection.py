from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

import numpy as np
from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.entrypoints.openai.realtime.connection import RealtimeConnection as VllmRealtimeConnection
from vllm.entrypoints.openai.realtime.protocol import TranscriptionDelta, TranscriptionDone
from vllm.logger import init_logger

from vllm_omni.entrypoints.async_omni import AsyncOmni

logger = init_logger(__name__)


class RealtimeConnection(VllmRealtimeConnection):
    """Omni realtime connection with audio-only server events.

    Reuses upstream vLLM websocket/session lifecycle and only customizes
    generation output handling to emit audio deltas.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Optional overrides from the client's session.update (see openai_realtime_client.py).
        self._session_sampling_patch: list[dict[str, Any]] | dict[str, Any] | None = None

    async def handle_event(self, event: dict):
        if event.get("type") == "session.update":
            self._capture_session_sampling_fields(event)
            if self._session_sampling_patch is not None:
                try:
                    self._build_realtime_sampling_params_list()
                except Exception as e:
                    logger.warning("Invalid realtime sampling params: %s", e)
                    await self.send_error(str(e), "invalid_sampling_params")
                    return
        await super().handle_event(event)

    def _capture_session_sampling_fields(self, event: dict) -> None:
        if "sampling_params_list" in event:
            val = event["sampling_params_list"]
            if val is None:
                self._session_sampling_patch = None
            elif not isinstance(val, list):
                raise ValueError("sampling_params_list must be a JSON array of objects or null")
            else:
                norm: list[dict[str, Any]] = []
                for item in val:
                    if not isinstance(item, dict):
                        raise ValueError("Each sampling_params_list entry must be a JSON object")
                    norm.append(dict(item))
                self._session_sampling_patch = norm
        elif "sampling_params" in event:
            val = event["sampling_params"]
            if val is None:
                self._session_sampling_patch = None
            elif not isinstance(val, dict):
                raise ValueError("sampling_params must be a JSON object or null")
            else:
                self._session_sampling_patch = dict(val)

    @staticmethod
    def _patch_sampling_params(base: Any, patch: dict[str, Any]) -> Any:
        out = base.clone()
        for key, value in patch.items():
            if not hasattr(out, key):
                continue
            setattr(out, key, value)
        return out

    def _build_realtime_sampling_params_list(self) -> list[Any]:
        patch = self._session_sampling_patch
        if patch is None:
            raise ValueError("No sampling parameter overrides are set for this session")
        ec = self.serving.engine_client
        defaults = ec.default_sampling_params_list
        out = [p.clone() for p in defaults]
        if isinstance(patch, dict):
            out[0] = self._patch_sampling_params(out[0], patch)
        else:
            if len(patch) != len(out):
                raise ValueError(
                    f"sampling_params_list must have one entry per pipeline stage "
                    f"(expected {len(out)}, got {len(patch)})"
                )
            for i, stage_patch in enumerate(patch):
                if not isinstance(stage_patch, dict):
                    raise ValueError("Each sampling_params_list entry must be a JSON object")
                out[i] = self._patch_sampling_params(out[i], stage_patch)
        AsyncOmni._validate_streaming_input_sampling_params(out[0])
        return out

    def _realtime_sampling_params_list_kw(self) -> dict[str, list[Any]]:
        if self._session_sampling_patch is None:
            return {}
        return {"sampling_params_list": self._build_realtime_sampling_params_list()}

    async def start_generation(self):
        await super().start_generation()

    @staticmethod
    def _tensor_to_numpy(value) -> np.ndarray | None:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            arr = value
        elif hasattr(value, "detach"):
            arr = value.detach().float().cpu().numpy()
        else:
            try:
                arr = np.asarray(value)
            except Exception:
                return None
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        return arr.astype(np.float32, copy=False)

    def _extract_audio_chunks(self, output) -> tuple[list[np.ndarray], int]:
        mm = getattr(output, "multimodal_output", None)
        if not isinstance(mm, dict):
            return [], 24000

        sr = mm.get("sr") or mm.get("sample_rate") or mm.get("audio_sample_rate") or 24000
        key = "audio" if "audio" in mm else ("model_outputs" if "model_outputs" in mm else None)
        if key is None:
            return [], int(sr)

        raw_audio = mm.get(key)
        chunks: list[np.ndarray] = []
        if isinstance(raw_audio, (list, tuple)):
            if len(raw_audio) > 0:
                arr = self._tensor_to_numpy(raw_audio[-1])
                if arr is not None and arr.size > 0:
                    chunks.append(arr)
        else:
            arr = self._tensor_to_numpy(raw_audio)
            if arr is not None and arr.size > 0:
                chunks.append(arr)
        return chunks, int(sr)

    @staticmethod
    def _pcm16_b64(audio_f32: np.ndarray) -> str:
        clipped = np.clip(audio_f32, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        return base64.b64encode(pcm16.tobytes()).decode("utf-8")

    async def _run_generation(
        self,
        streaming_input_gen: AsyncGenerator,
        input_stream: asyncio.Queue[list[int]],
    ):
        request_id = f"rt-{self.connection_id}-{uuid4()}"
        sent_audio = False
        audio_done_sent = False
        full_text = ""
        sent_text_len = 0
        prompt_token_ids_len = 0
        completion_tokens_len = 0

        try:
            result_gen = self.serving.engine_client.generate(
                prompt=streaming_input_gen,
                request_id=request_id,
                **self._realtime_sampling_params_list_kw(),
            )

            async for output in result_gen:
                if output.outputs and len(output.outputs) > 0:
                    output0 = output.outputs[0]
                    token_ids = list(output0.token_ids)
                    if token_ids:
                        input_stream.put_nowait(token_ids)
                        # token_ids are cumulative per request
                        completion_tokens_len = len(token_ids)
                    if not prompt_token_ids_len and output.prompt_token_ids:
                        prompt_token_ids_len = len(output.prompt_token_ids)
                    cumulative_text = output0.text or ""
                    if cumulative_text:
                        if len(cumulative_text) >= sent_text_len:
                            delta_text = cumulative_text[sent_text_len:]
                        else:
                            delta_text = cumulative_text
                        sent_text_len = len(cumulative_text)
                        full_text = cumulative_text
                    else:
                        delta_text = ""

                    if delta_text:
                        await self.send(TranscriptionDelta(delta=delta_text))

                audio_chunks, sample_rate = self._extract_audio_chunks(output)

                for chunk in audio_chunks:
                    sent_audio = True
                    await self.send_json(
                        {
                            "type": "response.audio.delta",
                            "audio": self._pcm16_b64(chunk),
                            "format": "pcm16",
                            "sample_rate_hz": sample_rate,
                        }
                    )

                if not self._is_connected:
                    break

            usage = UsageInfo(
                prompt_tokens=prompt_token_ids_len,
                completion_tokens=completion_tokens_len,
                total_tokens=prompt_token_ids_len + completion_tokens_len,
            )
            await self.send(TranscriptionDone(text=full_text, usage=usage))

            if sent_audio:
                await self.send_json({"type": "response.audio.done", "has_audio": True})
                audio_done_sent = True
        except Exception as e:
            logger.exception("Error in generation: %s", e)
            await self.send_error(str(e), "processing_error")
        finally:
            # Always send terminal event so clients don't hang forever.
            if self._is_connected and not audio_done_sent:
                try:
                    await self.send_json({"type": "response.audio.done", "has_audio": sent_audio})
                except Exception:
                    logger.exception("Failed to send response.audio.done")
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()

    async def send_json(self, payload: dict):
        await self.websocket.send_text(json.dumps(payload))

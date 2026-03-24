from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.sampling_params import SamplingParams

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.client_request_state import ClientRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
async def test_get_supported_tasks_returns_engine_supported_tasks():
    omni = object.__new__(AsyncOmni)
    omni.engine = SimpleNamespace(supported_tasks=("generate", "speech"))

    supported_tasks = await omni.get_supported_tasks()

    assert supported_tasks == ("generate", "speech")


def test_model_config_and_vllm_config_forward_from_comprehension_stage():
    model_config = SimpleNamespace(model="Qwen/Qwen3-TTS")
    vllm_config = SimpleNamespace(model_config=model_config)
    renderer = SimpleNamespace(name="renderer")
    input_processor = SimpleNamespace(renderer=renderer)
    io_processor = SimpleNamespace(name="io-processor")
    omni = object.__new__(AsyncOmni)
    omni.engine = SimpleNamespace(
        stage_clients=[SimpleNamespace(is_comprehension=False), SimpleNamespace(is_comprehension=True)],
        stage_vllm_configs=[None, vllm_config],
    )
    omni.input_processor = input_processor
    omni.io_processor = io_processor

    assert omni.vllm_config is vllm_config
    assert omni.model_config is model_config
    assert omni.renderer is renderer
    assert omni.input_processor is input_processor
    assert omni.io_processor is io_processor


def test_openai_serving_models_can_consume_async_omni_compat_attrs():
    model_config = SimpleNamespace(model="Qwen/Qwen3-TTS", max_model_len=32768)
    vllm_config = SimpleNamespace(model_config=model_config)
    renderer = SimpleNamespace(name="renderer")
    input_processor = SimpleNamespace(renderer=renderer)
    io_processor = SimpleNamespace(name="io-processor")
    omni = object.__new__(AsyncOmni)
    omni.engine = SimpleNamespace(
        stage_clients=[SimpleNamespace(is_comprehension=True)],
        stage_vllm_configs=[vllm_config],
    )
    omni.input_processor = input_processor
    omni.io_processor = io_processor

    serving_models = OpenAIServingModels(
        engine_client=omni,
        base_model_paths=[BaseModelPath(name="tts-model", model_path="Qwen/Qwen3-TTS")],
    )

    assert serving_models.model_config is model_config
    assert serving_models.renderer is renderer
    assert serving_models.io_processor is io_processor
    assert serving_models.input_processor is input_processor


@pytest.mark.asyncio
async def test_add_streaming_input_request_sends_updates_and_final_signal():
    omni = object.__new__(AsyncOmni)
    omni.engine = SimpleNamespace(
        add_request_async=AsyncMock(),
        add_streaming_update_async=AsyncMock(),
    )
    req_state = ClientRequestState("req-stream")
    omni.request_states = {"req-stream": req_state}

    params = SamplingParams(max_tokens=8)

    async def input_stream():
        yield SimpleNamespace(prompt={"prompt_token_ids": [1, 2, 3]}, sampling_params=None)
        yield SimpleNamespace(prompt={"prompt_token_ids": [4, 5]}, sampling_params=None)

    task = await omni._add_streaming_input_request(
        request_id="req-stream",
        input_stream=input_stream(),
        sampling_params_list=[params],
        final_stage_id=0,
    )
    await task

    assert omni.engine.add_request_async.await_count == 1
    first_call_kwargs = omni.engine.add_request_async.await_args.kwargs
    assert first_call_kwargs["request_id"] == "req-stream"
    assert first_call_kwargs["resumable"] is True

    assert omni.engine.add_streaming_update_async.await_count == 2
    stream_update_calls = omni.engine.add_streaming_update_async.await_args_list
    assert stream_update_calls[0].kwargs["resumable"] is True
    assert stream_update_calls[1].kwargs["resumable"] is False

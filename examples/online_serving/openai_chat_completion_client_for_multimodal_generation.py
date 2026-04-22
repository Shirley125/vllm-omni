"""OpenAI-compatible multimodal chat demo with optional benchmarking fields matching ``benchmark_realtime_ws.py``.

Streaming time anchor: clock starts when ``chat.completions.create(..., stream=True)`` returns
(stream open, request sent). **ttft_transcription_s** = first text delta; **ttfp_audio_s** = first audio delta.
Non-streaming runs only populate **e2e_s** / **rtf** / **tokens_per_s** (no first-token splits).
"""

from __future__ import annotations

import base64
import concurrent.futures
import io
import json
import math
import os
import statistics
import sys
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import requests
from openai import OpenAI
from vllm.assets.audio import AudioAsset
from vllm.utils.argparse_utils import FlexibleArgumentParser

SEED = 42


class QueryResult(NamedTuple):
    inputs: dict
    limit_mm_per_prompt: dict[str, int]


@dataclass
class RoundMetrics:
    """Per-request stats aligned with ``scripts/benchmark_realtime_ws.py`` field names.

    Time anchor (streaming): ``chat.completions.create(..., stream=True)`` has returned
    (HTTP request sent, stream open) — analogue to realtime ``final`` input commit.
    """

    client_id: int
    round_idx: int
    absolute_round: int
    ttft_transcription_s: float | None
    ttfp_audio_s: float | None
    e2e_s: float | None
    rtf: float | None
    tokens_per_s: float | None
    completion_tokens: int
    output_audio_seconds: float
    output_sample_rate_hz: int
    error: str | None = None


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return math.nan
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(math.floor(k))
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


def _fmt_stats(values: list[float]) -> str:
    xs = sorted(v for v in values if not math.isnan(v))
    if not xs:
        return "n=0"
    return (
        f"n={len(xs)} mean={statistics.mean(xs):.4f}s "
        f"p50={_percentile(xs, 50):.4f}s p95={_percentile(xs, 95):.4f}s p99={_percentile(xs, 99):.4f}s"
    )


def _wav_duration_and_sr(audio_bytes: bytes) -> tuple[float, int]:
    """Return (duration_seconds, sample_rate_hz) if ``audio_bytes`` is a WAV blob; else (0, 0)."""
    if not audio_bytes:
        return 0.0, 0
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            sr = wf.getframerate()
            if sr <= 0:
                return 0.0, 0
            dur = wf.getnframes() / float(sr)
            return dur, int(sr)
    except Exception:
        return 0.0, 0


def _print_metrics_summary(rows: list[RoundMetrics]) -> None:
    ok = [r for r in rows if r.error is None]
    bad = [r for r in rows if r.error is not None]

    print(f"[summary] rounds_recorded={len(rows)} ok={len(ok)} failed={len(bad)}")
    if bad:
        for r in bad[:20]:
            print(f"  client={r.client_id} abs_round={r.absolute_round} err={r.error}")
        if len(bad) > 20:
            print(f"  ... and {len(bad) - 20} more failures")

    if not ok:
        return

    def col(name: str, getter) -> None:
        vals = [getter(r) for r in ok]
        vals_f = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
        print(f"  {name}: {_fmt_stats(vals_f)}")

    col("ttft_transcription_s", lambda r: r.ttft_transcription_s)
    col("ttfp_audio_s", lambda r: r.ttfp_audio_s)
    col("e2e_s", lambda r: r.e2e_s)
    col("rtf", lambda r: r.rtf)
    col("tokens_per_s", lambda r: r.tokens_per_s)

    total_e2e = sum(r.e2e_s for r in ok if r.e2e_s is not None)
    total_tokens = sum(r.completion_tokens for r in ok)
    total_audio_s = sum(r.output_audio_seconds for r in ok)
    if total_e2e > 0:
        print(
            f"  aggregate_tokens_per_s (sum_tokens/sum_e2e): {total_tokens / total_e2e:.4f} "
            f"(sum_completion_tokens={total_tokens}, sum_e2e_s={total_e2e:.4f})"
        )
    if total_e2e > 0 and total_audio_s > 0:
        print(f"  aggregate_rtf (sum_e2e/sum_output_audio_s): {total_e2e / total_audio_s:.4f}")


def make_audio_output_filename(request_id: str | None, index: int) -> str:
    """Build a stable output filename using request ID when available."""
    if not request_id:
        request_id = f"unknown_{index}"
    safe_request_id = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in request_id)
    return f"audio_{safe_request_id}_{index}.wav"


def encode_base64_content_from_url(content_url: str) -> str:
    """Encode a content retrieved from a remote url to base64 format."""

    with requests.get(content_url) as response:
        response.raise_for_status()
        result = base64.b64encode(response.content).decode("utf-8")

    return result


def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a local file to base64 format."""
    with open(file_path, "rb") as f:
        content = f.read()
        result = base64.b64encode(content).decode("utf-8")
    return result


def get_video_url_from_path(video_path: str | None) -> str:
    """Convert a video path (local file or URL) to a video URL format for the API.

    If video_path is None or empty, returns the default URL.
    If video_path is a local file path, encodes it to base64 data URL.
    If video_path is a URL, returns it as-is.
    """
    if not video_path:
        # Default video URL
        return "https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4"

    # Check if it's a URL (starts with http:// or https://)
    if video_path.startswith(("http://", "https://")):
        return video_path

    # Otherwise, treat it as a local file path
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Detect video MIME type from file extension
    video_path_lower = video_path.lower()
    if video_path_lower.endswith(".mp4"):
        mime_type = "video/mp4"
    elif video_path_lower.endswith(".webm"):
        mime_type = "video/webm"
    elif video_path_lower.endswith(".mov"):
        mime_type = "video/quicktime"
    elif video_path_lower.endswith(".avi"):
        mime_type = "video/x-msvideo"
    elif video_path_lower.endswith(".mkv"):
        mime_type = "video/x-matroska"
    else:
        # Default to mp4 if extension is unknown
        mime_type = "video/mp4"

    video_base64 = encode_base64_content_from_file(video_path)
    return f"data:{mime_type};base64,{video_base64}"


def get_image_url_from_path(image_path: str | None) -> str:
    """Convert an image path (local file or URL) to an image URL format for the API.

    If image_path is None or empty, returns the default URL.
    If image_path is a local file path, encodes it to base64 data URL.
    If image_path is a URL, returns it as-is.
    """
    if not image_path:
        # Default image URL
        return "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg"

    # Check if it's a URL (starts with http:// or https://)
    if image_path.startswith(("http://", "https://")):
        return image_path

    # Otherwise, treat it as a local file path
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Detect image MIME type from file extension
    image_path_lower = image_path.lower()
    if image_path_lower.endswith((".jpg", ".jpeg")):
        mime_type = "image/jpeg"
    elif image_path_lower.endswith(".png"):
        mime_type = "image/png"
    elif image_path_lower.endswith(".gif"):
        mime_type = "image/gif"
    elif image_path_lower.endswith(".webp"):
        mime_type = "image/webp"
    else:
        # Default to jpeg if extension is unknown
        mime_type = "image/jpeg"

    image_base64 = encode_base64_content_from_file(image_path)
    return f"data:{mime_type};base64,{image_base64}"


def get_audio_url_from_path(audio_path: str | None) -> str:
    """Convert an audio path (local file or URL) to an audio URL format for the API.

    If audio_path is None or empty, returns the default URL.
    If audio_path is a local file path, encodes it to base64 data URL.
    If audio_path is a URL, returns it as-is.
    """
    if not audio_path:
        # Default audio URL
        return AudioAsset("mary_had_lamb").url

    # Check if it's a URL (starts with http:// or https://)
    if audio_path.startswith(("http://", "https://")):
        return audio_path

    # Otherwise, treat it as a local file path
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Detect audio MIME type from file extension
    audio_path_lower = audio_path.lower()
    if audio_path_lower.endswith((".mp3", ".mpeg")):
        mime_type = "audio/mpeg"
    elif audio_path_lower.endswith(".wav"):
        mime_type = "audio/wav"
    elif audio_path_lower.endswith(".ogg"):
        mime_type = "audio/ogg"
    elif audio_path_lower.endswith(".flac"):
        mime_type = "audio/flac"
    elif audio_path_lower.endswith(".m4a"):
        mime_type = "audio/mp4"
    else:
        # Default to wav if extension is unknown
        mime_type = "audio/wav"

    audio_base64 = encode_base64_content_from_file(audio_path)
    return f"data:{mime_type};base64,{audio_base64}"


def get_system_prompt():
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are Qwen, a virtual human developed by the Qwen Team, "
                    "Alibaba Group, capable of perceiving auditory and visual inputs, "
                    "as well as generating text and speech."
                ),
            }
        ],
    }


def _parse_csv_arg(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _build_prompt_for_query_type(
    query_type: str,
    custom_prompt: str | None,
    video_path: str | None,
    image_path: str | None,
    audio_path: str | None,
):
    query_func = query_map[query_type]
    if query_type == "use_video":
        return query_func(video_path=video_path, custom_prompt=custom_prompt)
    if query_type == "use_image":
        return query_func(image_path=image_path, custom_prompt=custom_prompt)
    if query_type == "use_audio":
        return query_func(audio_path=audio_path, custom_prompt=custom_prompt)
    if query_type == "text":
        return query_func(custom_prompt=custom_prompt)
    if query_type == "use_audio_in_video":
        return query_func(video_path=video_path, custom_prompt=custom_prompt)
    # use_mixed_modalities / use_multi_audios
    return query_func(custom_prompt=custom_prompt)


def get_text_query(custom_prompt: str | None = None):
    question = (
        custom_prompt or "Explain the system architecture for a scalable audio generation pipeline. Answer in 15 words."
    )
    prompt = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"{question}",
            }
        ],
    }
    return prompt


default_system = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
    "Group, capable of perceiving auditory and visual inputs, as well as "
    "generating text and speech."
)


def get_video_query(video_path: str | None = None, custom_prompt: str | None = None):
    question = custom_prompt or "Why is this video funny?"
    video_url = get_video_url_from_path(video_path)
    prompt = {
        "role": "user",
        "content": [
            {
                "type": "video_url",
                "video_url": {"url": video_url},
            },
            {
                "type": "text",
                "text": f"{question}",
            },
        ],
    }
    return prompt


def get_image_query(image_path: str | None = None, custom_prompt: str | None = None):
    question = custom_prompt or "What is the content of this image?"
    image_url = get_image_url_from_path(image_path)
    prompt = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": image_url},
            },
            {
                "type": "text",
                "text": f"{question}",
            },
        ],
    }
    return prompt


def get_audio_query(audio_path: str | None = None, custom_prompt: str | None = None):
    question = custom_prompt or "What is the content of this audio?"
    audio_url = get_audio_url_from_path(audio_path)
    prompt = {
        "role": "user",
        "content": [
            {
                "type": "audio_url",
                "audio_url": {"url": audio_url},
            },
            {
                "type": "text",
                "text": f"{question}",
            },
        ],
    }
    return prompt


def get_mixed_modalities_query(
    video_path: str | None = None,
    image_path: str | None = None,
    audio_path: str | None = None,
    custom_prompt: str | None = None,
):
    """
    Online-friendly multimodal user message:
    - Uses URLs (or base64 data URLs) for audio / image / video.
    - Returns the OpenAI-style message dict directly (not the offline QueryResult).
    """
    question = (
        custom_prompt or "What is recited in the audio? What is the content of this image? Why is this video funny?"
    )

    audio_url = get_audio_url_from_path(audio_path)
    image_url = get_image_url_from_path(image_path)
    video_url = get_video_url_from_path(video_path)

    return {
        "role": "user",
        "content": [
            {"type": "audio_url", "audio_url": {"url": audio_url}},
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "video_url", "video_url": {"url": video_url}},
            {"type": "text", "text": question},
        ],
    }


def get_multi_audios_query(custom_prompt: str | None = None):
    """
    Online-friendly two-audio comparison request.
    - Encodes both audio clips as URLs (or data URLs).
    - Returns the OpenAI-style message dict.
    """
    question = custom_prompt or "Are these two audio clips the same?"
    # Use default demo clips; you can point to your own via --audio-path if needed.
    audio_url_1 = get_audio_url_from_path(AudioAsset("winning_call").url)
    audio_url_2 = get_audio_url_from_path(AudioAsset("mary_had_lamb").url)

    return {
        "role": "user",
        "content": [
            {"type": "audio_url", "audio_url": {"url": audio_url_1}},
            {"type": "audio_url", "audio_url": {"url": audio_url_2}},
            {"type": "text", "text": question},
        ],
    }


def get_use_audio_in_video_query(
    video_path: str | None = None,
    custom_prompt: str | None = None,
):
    """Query for use_audio_in_video mode.

    When use_audio_in_video=True, audio is automatically extracted from the video
    by the server. Do NOT send a separate audio_url - this would cause a mismatch
    between the number of audio and video items.
    """
    question = custom_prompt or (
        "Describe the content of the video in details, then convert what the baby say into text."
    )
    video_url = get_video_url_from_path(video_path)
    # Note: audio is extracted from video automatically when use_audio_in_video=True
    # Do not include a separate audio_url here
    return {
        "role": "user",
        "content": [
            {"type": "video_url", "video_url": {"url": video_url}},
            {"type": "text", "text": question},
        ],
    }


query_map = {
    "text": get_text_query,
    "use_audio": get_audio_query,
    "use_image": get_image_query,
    "use_video": get_video_query,
    "use_mixed_modalities": get_mixed_modalities_query,
    "use_multi_audios": get_multi_audios_query,
    "use_audio_in_video": get_use_audio_in_video_query,
}


def _messages_for_payload(payload: dict) -> list:
    return [get_system_prompt(), payload["prompt"]]


def _run_single_round(
    *,
    client: OpenAI,
    model_name: str,
    payload: dict,
    output_modalities: list[str] | None,
    stream: bool,
    client_id: int,
    absolute_round: int,
    round_idx: int,
    is_warmup: bool,
    audio_file_index_start: int,
    save_audio_files: bool,
) -> tuple[RoundMetrics, int]:
    """One chat completion; metrics match :mod:`benchmark_realtime_ws` naming.

    Returns ``(metrics, next_audio_file_index)`` for sequential filenames across rounds.
    """
    err: str | None = None
    ttft_t: float | None = None
    ttfp_a: float | None = None
    e2e: float | None = None
    rtf: float | None = None
    tok_ps: float | None = None
    completion_tokens = 0
    audio_accum = bytearray()
    out_seconds = 0.0
    out_sr = 0
    file_idx = audio_file_index_start

    try:
        if not stream:
            t0 = time.perf_counter()
            chat_completion = client.chat.completions.create(
                messages=_messages_for_payload(payload),
                model=model_name,
                modalities=output_modalities,
                extra_body=payload["extra_body"],
                stream=False,
            )
            t1 = time.perf_counter()
            e2e = t1 - t0
            usage = getattr(chat_completion, "usage", None)
            if usage is not None:
                ct = getattr(usage, "completion_tokens", None)
                if isinstance(ct, int):
                    completion_tokens = ct
            request_id = getattr(chat_completion, "id", None)
            for choice in chat_completion.choices:
                if choice.message.audio:
                    audio_data = base64.b64decode(choice.message.audio.data)
                    audio_accum.extend(audio_data)
                    if save_audio_files and not is_warmup:
                        audio_file_path = make_audio_output_filename(request_id=request_id, index=file_idx)
                        with open(audio_file_path, "wb") as f:
                            f.write(audio_data)
                        print(f"Audio saved to {audio_file_path}")
                        file_idx += 1
                elif choice.message.content and not is_warmup:
                    print("Chat completion output from text:", choice.message.content)
        else:
            stream_it = client.chat.completions.create(
                messages=_messages_for_payload(payload),
                model=model_name,
                modalities=output_modalities,
                extra_body=payload["extra_body"],
                stream=True,
            )
            t_after_create = time.perf_counter()
            first_text_t: float | None = None
            first_audio_t: float | None = None
            printed_content = False
            for chunk in stream_it:
                now = time.perf_counter()
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    ct = getattr(usage, "completion_tokens", None)
                    if isinstance(ct, int):
                        completion_tokens = ct
                modality = getattr(chunk, "modality", None)
                for choice in chunk.choices:
                    if hasattr(choice, "delta"):
                        content = getattr(choice.delta, "content", None)
                    else:
                        content = None

                    if modality == "audio" and content:
                        audio_data = base64.b64decode(content)
                        audio_accum.extend(audio_data)
                        if first_audio_t is None:
                            first_audio_t = now
                        if save_audio_files and not is_warmup:
                            request_id = getattr(chunk, "id", None)
                            audio_file_path = make_audio_output_filename(request_id=request_id, index=file_idx)
                            with open(audio_file_path, "wb") as f:
                                f.write(audio_data)
                            print(f"\nAudio saved to {audio_file_path}")
                            file_idx += 1
                    elif modality == "text" and content:
                        if first_text_t is None:
                            first_text_t = now
                        if not is_warmup:
                            if not printed_content:
                                printed_content = True
                                print("\ncontent:", end="", flush=True)
                            print(content, end="", flush=True)

            t_end = time.perf_counter()
            e2e = t_end - t_after_create
            if first_text_t is not None:
                ttft_t = first_text_t - t_after_create
            if first_audio_t is not None:
                ttfp_a = first_audio_t - t_after_create

        dur, sr = _wav_duration_and_sr(bytes(audio_accum))
        if dur > 0:
            out_seconds = dur
            out_sr = sr

        if err is None and e2e is not None and e2e > 0 and out_seconds > 0:
            rtf = e2e / out_seconds
        if err is None and e2e is not None and e2e > 0 and completion_tokens > 0:
            tok_ps = completion_tokens / e2e

    except Exception as exc:
        err = str(exc)
        ttft_t = ttfp_a = e2e = rtf = tok_ps = None
        out_seconds = 0.0
        out_sr = 0
        completion_tokens = 0
        file_idx = audio_file_index_start

    metrics = RoundMetrics(
        client_id=client_id,
        round_idx=round_idx,
        absolute_round=absolute_round,
        ttft_transcription_s=ttft_t if err is None else None,
        ttfp_audio_s=ttfp_a if err is None else None,
        e2e_s=e2e if err is None else None,
        rtf=rtf if err is None else None,
        tokens_per_s=tok_ps if err is None else None,
        completion_tokens=completion_tokens,
        output_audio_seconds=out_seconds,
        output_sample_rate_hz=out_sr,
        error=err,
    )
    return metrics, file_idx


def _worker_sequential_rounds(
    client_id: int,
    payload: dict,
    args,
    client: OpenAI,
    output_modalities: list[str] | None,
) -> list[RoundMetrics]:
    total_rounds = args.warmup_rounds + args.rounds
    results: list[RoundMetrics] = []
    audio_idx = client_id * 100000
    for r in range(1, total_rounds + 1):
        is_warmup = r <= args.warmup_rounds
        counted_idx = r - args.warmup_rounds
        m, audio_idx = _run_single_round(
            client=client,
            model_name=args.model,
            payload=payload,
            output_modalities=output_modalities,
            stream=args.stream,
            client_id=client_id,
            absolute_round=r,
            round_idx=counted_idx if not is_warmup else -1,
            is_warmup=is_warmup,
            audio_file_index_start=audio_idx,
            save_audio_files=not getattr(args, "no_save_audio", False),
        )
        if not is_warmup:
            results.append(m)
        if m.error:
            break
    return results


def run_multimodal_generation(args, client: OpenAI) -> list[RoundMetrics]:
    if args.rounds < 1:
        raise ValueError("--rounds must be >= 1")
    if args.warmup_rounds < 0:
        raise ValueError("--warmup-rounds must be >= 0")

    # Get paths and custom prompt from args
    video_path = getattr(args, "video_path", None)
    image_path = getattr(args, "image_path", None)
    audio_path = getattr(args, "audio_path", None)
    custom_prompt = getattr(args, "prompt", None)

    if args.modalities is not None:
        output_modalities = args.modalities.split(",")
    else:
        output_modalities = None

    num_concurrent_requests = args.num_concurrent_requests
    prompt_list = _parse_csv_arg(getattr(args, "prompts", None))
    speaker_list = _parse_csv_arg(getattr(args, "speakers", None))

    request_payloads = []
    for idx in range(num_concurrent_requests):
        per_req_prompt = (
            prompt_list[idx]
            if idx < len(prompt_list)
            else (custom_prompt if idx == 0 or not prompt_list else prompt_list[-1])
        )
        per_req_speaker = (
            speaker_list[idx]
            if idx < len(speaker_list)
            else (args.speaker if idx == 0 or not speaker_list else speaker_list[-1])
        )
        prompt = _build_prompt_for_query_type(
            query_type=args.query_type,
            custom_prompt=per_req_prompt,
            video_path=video_path,
            image_path=image_path,
            audio_path=audio_path,
        )
        extra_body = {
            # Optional, it has default settings in stage configs. you can override them here.
        }
        if args.query_type == "use_audio_in_video":
            extra_body["mm_processor_kwargs"] = {"use_audio_in_video": True}
        if per_req_speaker and per_req_speaker.strip():
            extra_body["speaker"] = per_req_speaker.strip()
        request_payloads.append({"prompt": prompt, "extra_body": extra_body})

    all_metrics: list[RoundMetrics] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
        futures = [
            executor.submit(
                _worker_sequential_rounds,
                idx + 1,
                request_payloads[idx],
                args,
                client,
                output_modalities,
            )
            for idx in range(num_concurrent_requests)
        ]
        for fut in futures:
            all_metrics.extend(fut.result())

    jsonl_out = getattr(args, "jsonl_out", None)
    if jsonl_out:
        out_path = Path(jsonl_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for m in all_metrics:
                f.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")

    _print_metrics_summary(all_metrics)
    return all_metrics


def parse_args():
    parser = FlexibleArgumentParser(description="Demo on using vLLM for offline inference with audio language models")
    parser.add_argument(
        "--query-type",
        "-q",
        type=str,
        default="use_audio_in_video",
        choices=query_map.keys(),
        help="Query type.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Model Name / Path",
    )
    parser.add_argument(
        "--video-path",
        "-v",
        type=str,
        default=None,
        help="Path to local video file or URL. If not provided and query-type is 'use_video', uses default video URL.",
    )
    parser.add_argument(
        "--image-path",
        "-i",
        type=str,
        default=None,
        help="Path to local image file or URL. If not provided and query-type is 'use_image', uses default image URL.",
    )
    parser.add_argument(
        "--audio-path",
        "-a",
        type=str,
        default=None,
        help="Path to local audio file or URL. If not provided and query-type is 'use_audio', uses default audio URL.",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default=None,
        help="Custom text prompt/question to use instead of the default prompt for the selected query type.",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default=None,
        help="Output modalities to use for the prompts.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the response.",
    )
    parser.add_argument(
        "--num-concurrent-requests",
        type=int,
        default=1,
        help="Number of concurrent workers; each runs warmup + rounds sequentially.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Counted completion rounds per worker after warmup (metrics / JSONL).",
    )
    parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=0,
        help="Rounds per worker excluded from metrics (same idea as benchmark_realtime_ws.py).",
    )
    parser.add_argument(
        "--jsonl-out",
        type=str,
        default=None,
        help="Write one JSON line per recorded round (RoundMetrics fields).",
    )
    parser.add_argument(
        "--no-save-audio",
        action="store_true",
        help="Do not write per-chunk WAV files; still decode accumulated audio for RTF when possible.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8091,
        help="Port of the vLLM Omni API server.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host/IP of the vLLM Omni API server.",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default=None,
        help="TTS speaker/voice for audio output (e.g. Ethan, Vivian). Passed via extra_body to the talker stage.",
    )
    parser.add_argument(
        "--speakers",
        type=str,
        default=None,
        help=(
            "Comma-separated speakers for concurrent requests, e.g. "
            "'Ethan,Vivian,Ryan'. Overrides --speaker per request."
        ),
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help=(
            "Comma-separated prompts for concurrent requests. "
            "If fewer than --num-concurrent-requests, the last prompt is reused."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    host = args.host
    port = args.port
    openai_api_base = f"http://{host}:{port}/v1"
    client = OpenAI(
        api_key="EMPTY",
        base_url=openai_api_base,
    )
    metrics = run_multimodal_generation(args, client)
    if any(m.error for m in metrics):
        sys.exit(1)

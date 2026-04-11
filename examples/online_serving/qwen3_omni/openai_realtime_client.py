"""Realtime audio client for vLLM-Omni /v1/realtime (audio-only events).

This client:
1) Reads a local WAV file (must be mono, 16-bit PCM, 16kHz),
2) Streams PCM16 chunks to /v1/realtime with OpenAI-style events,
3) Receives response.audio.delta / response.audio.done,
4) Saves synthesized audio to an output WAV file.

By default each ``response.audio.delta`` is treated as a **full cumulative PCM**
snapshot (common for vLLM-Omni): only the **last** delta is written to
``--output-wav``. Use ``--join-audio-deltas`` if your server sends true
**incremental** PCM chunks (concatenate every delta).

Optional debugging: pass ``--delta-dump-dir DIR`` to write every
``response.audio.delta`` payload as ``delta_000001.wav``, ``delta_000002.wav``, …

Usage:
  python openai_realtime_client.py \
      --url ws://localhost:8091/v1/realtime \
      --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
      --input-wav input_16k_mono.wav \
      --output-wav realtime_output.wav \
      --delta-dump-dir ./rt_delta_wavs

Dependencies:
  pip install websockets
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import wave
from pathlib import Path

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets")
    raise SystemExit(1)


def _read_wav_pcm16(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        comptype = wf.getcomptype()
        nframes = wf.getnframes()

        if nchannels != 1:
            raise ValueError(f"Input WAV must be mono (got {nchannels} channels).")
        if sampwidth != 2:
            raise ValueError(f"Input WAV must be 16-bit PCM (got sample width={sampwidth}).")
        if framerate != 16000:
            raise ValueError(f"Input WAV must be 16kHz (got {framerate} Hz).")
        if comptype != "NONE":
            raise ValueError(f"Input WAV must be uncompressed PCM (got comptype={comptype}).")
        if nframes <= 0:
            raise ValueError("Input WAV has no audio frames.")

        return wf.readframes(nframes)


def _write_wav_pcm16(path: Path, pcm16_bytes: bytes, sample_rate_hz: int) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate_hz)
        wf.writeframes(pcm16_bytes)


async def run_client(
    url: str,
    model: str,
    input_wav: Path,
    output_wav: Path,
    chunk_ms: int,
    send_delay_ms: int,
    delta_dump_dir: Path | None,
    join_audio_deltas: bool,
) -> None:
    pcm16 = _read_wav_pcm16(input_wav)
    bytes_per_ms = 16000 * 2 // 1000  # mono PCM16 at 16kHz
    chunk_bytes = max(bytes_per_ms * chunk_ms, 2)

    latest_full_pcm: bytes | None = None
    incremental_pcm_parts: list[bytes] = []
    output_sample_rate = 24000
    delta_index = 0

    if delta_dump_dir is not None:
        delta_dump_dir.mkdir(parents=True, exist_ok=True)

    async with websockets.connect(url, max_size=64 * 1024 * 1024) as ws:
        # 1) Validate model.
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "model": model,
                }
            )
        )

        # 2) Start generation once (non-final commit).
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": False}))

        # 3) Stream audio chunks.
        for i in range(0, len(pcm16), chunk_bytes):
            chunk = pcm16[i : i + chunk_bytes]
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("utf-8"),
                    }
                )
            )
            if send_delay_ms > 0:
                await asyncio.sleep(send_delay_ms / 1000.0)

        # 4) Final commit closes input stream.
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))

        # 5) Receive server events until audio done.
        while True:
            message = await ws.recv()
            if isinstance(message, bytes):
                # We only expect JSON text frames.
                continue

            event = json.loads(message)
            event_type = event.get("type")

            if event_type == "session.created":
                continue

            if event_type == "response.audio.delta":
                sr = event.get("sample_rate_hz")
                if isinstance(sr, int) and sr > 0:
                    output_sample_rate = sr
                audio_b64 = event.get("audio", "")
                if audio_b64:
                    pcm_delta = base64.b64decode(audio_b64)
                    if join_audio_deltas:
                        incremental_pcm_parts.append(pcm_delta)
                    else:
                        latest_full_pcm = pcm_delta
                    if delta_dump_dir is not None and pcm_delta:
                        delta_index += 1
                        dump_path = delta_dump_dir / f"delta_{delta_index:06d}.wav"
                        _write_wav_pcm16(dump_path, pcm_delta, output_sample_rate)
                        print(
                            f"delta dump #{delta_index}: {dump_path} "
                            f"(pcm bytes={len(pcm_delta)}, sr={output_sample_rate})"
                        )
                continue

            if event_type == "response.audio.done":
                break

            if event_type == "error":
                raise RuntimeError(f"Server error: {event}")

        if join_audio_deltas:
            all_pcm16 = b"".join(incremental_pcm_parts)
        else:
            all_pcm16 = latest_full_pcm or b""
        if not all_pcm16:
            raise RuntimeError("No audio received from server.")

        output_wav.parent.mkdir(parents=True, exist_ok=True)
        _write_wav_pcm16(output_wav, all_pcm16, output_sample_rate)
        mode = "joined increments" if join_audio_deltas else "last cumulative delta"
        print(f"Saved realtime audio to: {output_wav} ({mode})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime audio client for vLLM-Omni")
    parser.add_argument("--url", default="ws://localhost:8091/v1/realtime", help="WebSocket URL")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Model name for session.update",
    )
    parser.add_argument("--input-wav", required=True, type=Path, help="Input WAV (mono, PCM16, 16kHz)")
    parser.add_argument("--output-wav", default=Path("realtime_output.wav"), type=Path, help="Output WAV path")
    parser.add_argument("--chunk-ms", type=int, default=200, help="Input chunk size in milliseconds")
    parser.add_argument(
        "--send-delay-ms",
        type=int,
        default=0,
        help="Delay between chunk sends; set >0 to simulate realtime upload",
    )
    parser.add_argument(
        "--delta-dump-dir",
        type=Path,
        default=None,
        help="If set, each response.audio.delta is saved as delta_NNNNNN.wav under this directory",
    )
    parser.add_argument(
        "--join-audio-deltas",
        action="store_true",
        help="Concatenate every delta as incremental PCM (default: each delta is full audio; keep last only)",
    )
    args = parser.parse_args()

    asyncio.run(
        run_client(
            url=args.url,
            model=args.model,
            input_wav=args.input_wav,
            output_wav=args.output_wav,
            chunk_ms=args.chunk_ms,
            send_delay_ms=args.send_delay_ms,
            delta_dump_dir=args.delta_dump_dir,
            join_audio_deltas=args.join_audio_deltas,
        )
    )


if __name__ == "__main__":
    main()

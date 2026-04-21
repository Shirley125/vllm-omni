#!/usr/bin/env python3
"""Benchmark WebSocket ``/v1/realtime`` with streaming PCM16 input (OpenAI-style events).

Mirrors ``examples/online_serving/qwen3_omni/openai_realtime_client.py`` chunking and flow.
Each *connection* runs multiple *rounds* (repeat append → final commit → drain until
``response.audio.done``).

Metrics (per round, anchored at ``input_audio_buffer.commit`` with ``final: true``):

- **ttft_transcription_s**: time to first ``transcription.delta`` (non-empty delta).
- **ttfp_audio_s**: time to first ``response.audio.delta`` with audio payload.
- **e2e_s**: wall time until ``response.audio.done`` for that round.
- **rtf**: ``e2e_s / output_audio_duration_s`` (processing time vs output audio length; <1 is faster than real-time).
- **tokens_per_s**: ``completion_tokens / e2e_s`` from ``transcription.done`` usage.

Usage:
  python scripts/benchmark_realtime_ws.py \\
      --url ws://localhost:8091/v1/realtime \\
      --model Qwen/Qwen3-Omni-30B-A3B-Instruct \\
      --input-wav path/to/input_16k_mono.wav \\
      --concurrency 4 \\
      --connections 8 \\
      --rounds 3 \\
      --warmup-rounds 1

Dependencies:
  pip install websockets
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import statistics
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import websockets
except ImportError as e:
    raise SystemExit("Please install websockets: pip install websockets") from e


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
            raise ValueError(f"Input WAV must be 16-bit PCM (got sampwidth={sampwidth}).")
        if framerate != 16000:
            raise ValueError(f"Input WAV must be 16kHz (got {framerate} Hz).")
        if comptype != "NONE":
            raise ValueError(f"Input WAV must be uncompressed PCM (got comptype={comptype}).")
        if nframes <= 0:
            raise ValueError("Input WAV has no audio frames.")

        return wf.readframes(nframes)


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


@dataclass
class RoundMetrics:
    client_id: int
    round_idx: int  # 1-based among counted rounds (after warmup), or absolute index for logging
    absolute_round: int  # 1-based including warmup
    ttft_transcription_s: float | None
    ttfp_audio_s: float | None
    e2e_s: float | None
    rtf: float | None
    tokens_per_s: float | None
    completion_tokens: int
    output_audio_seconds: float
    output_sample_rate_hz: int
    error: str | None = None


async def _stream_one_round(
    ws,
    pcm16: bytes,
    *,
    chunk_bytes: int,
    send_delay_s: float,
) -> None:
    await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": False}))
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
        if send_delay_s > 0:
            await asyncio.sleep(send_delay_s)
    await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))


async def run_client_benchmark(
    *,
    client_id: int,
    url: str,
    model: str,
    pcm16: bytes,
    chunk_ms: int,
    send_delay_ms: int,
    rounds: int,
    warmup_rounds: int,
    recv_timeout_s: float,
    max_size: int,
) -> list[RoundMetrics]:
    bytes_per_ms = 16000 * 2 // 1000
    chunk_bytes = max(bytes_per_ms * chunk_ms, 2)
    send_delay_s = send_delay_ms / 1000.0

    results: list[RoundMetrics] = []

    async with websockets.connect(url, max_size=max_size) as ws:
        await ws.send(json.dumps({"type": "session.update", "model": model}))

        for r in range(1, warmup_rounds + rounds + 1):
            is_warmup = r <= warmup_rounds
            counted_idx = r - warmup_rounds  # 1..rounds when not warmup

            t_final_commit = None
            first_transcription_t: float | None = None
            first_audio_t: float | None = None
            t_audio_done: float | None = None
            incremental: list[bytes] = []
            output_sr = 24000
            completion_tokens = 0
            err: str | None = None

            try:
                await _stream_one_round(
                    ws,
                    pcm16,
                    chunk_bytes=chunk_bytes,
                    send_delay_s=send_delay_s,
                )
                t_final_commit = time.perf_counter()

                while True:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=recv_timeout_s)
                    except asyncio.TimeoutError:
                        err = f"recv timeout after {recv_timeout_s}s"
                        break

                    if isinstance(message, bytes):
                        continue

                    event = json.loads(message)
                    event_type = event.get("type")
                    now = time.perf_counter()

                    if event_type == "session.created":
                        continue

                    if event_type == "response.audio.delta":
                        sr = event.get("sample_rate_hz")
                        if isinstance(sr, int) and sr > 0:
                            output_sr = sr
                        audio_b64 = event.get("audio", "")
                        if audio_b64:
                            pcm_delta = base64.b64decode(audio_b64)
                            incremental.append(pcm_delta)
                            if first_audio_t is None and t_final_commit is not None:
                                first_audio_t = now
                        continue

                    if event_type == "transcription.delta":
                        delta = event.get("delta", "")
                        if delta and first_transcription_t is None and t_final_commit is not None:
                            first_transcription_t = now
                        continue

                    if event_type == "transcription.done":
                        usage = event.get("usage") or {}
                        ct = usage.get("completion_tokens")
                        if isinstance(ct, int):
                            completion_tokens = ct
                        continue

                    if event_type == "response.audio.done":
                        t_audio_done = now
                        break

                    if event_type == "error":
                        err = f"server error: {event}"
                        break

                    # Ignore unknown event types (protocol extensions) without failing the benchmark.
                    continue

                if err is None and t_final_commit is None:
                    err = "internal: missing final commit timestamp"
                if err is None and t_audio_done is None:
                    err = "missing response.audio.done"

                ttft_t: float | None = None
                ttfp_a: float | None = None
                e2e: float | None = None
                rtf: float | None = None
                tok_ps: float | None = None
                out_seconds = 0.0

                if err is None:
                    out_pcm = b"".join(incremental)
                    out_seconds = len(out_pcm) / (2 * max(output_sr, 1))

                    ttft_t = (
                        first_transcription_t - t_final_commit
                        if first_transcription_t is not None and t_final_commit is not None
                        else None
                    )
                    ttfp_a = (
                        first_audio_t - t_final_commit
                        if first_audio_t is not None and t_final_commit is not None
                        else None
                    )
                    e2e = t_audio_done - t_final_commit if t_audio_done and t_final_commit else None

                    if e2e is not None and e2e > 0 and out_seconds > 0:
                        rtf = e2e / out_seconds

                    if e2e is not None and e2e > 0 and completion_tokens > 0:
                        tok_ps = completion_tokens / e2e

            except Exception as exc:
                err = str(exc)
                ttft_t = ttfp_a = e2e = rtf = tok_ps = None
                out_seconds = 0.0
                output_sr = 24000
                completion_tokens = 0

            m = RoundMetrics(
                client_id=client_id,
                round_idx=counted_idx if not is_warmup else -1,
                absolute_round=r,
                ttft_transcription_s=ttft_t if err is None else None,
                ttfp_audio_s=ttfp_a if err is None else None,
                e2e_s=e2e if err is None else None,
                rtf=rtf if err is None else None,
                tokens_per_s=tok_ps if err is None else None,
                completion_tokens=completion_tokens,
                output_audio_seconds=out_seconds,
                output_sample_rate_hz=output_sr,
                error=err,
            )
            if not is_warmup:
                results.append(m)
            if err:
                # Do not continue further rounds after failure
                break

    return results


async def run_benchmark(
    *,
    url: str,
    model: str,
    input_wav: Path,
    chunk_ms: int,
    send_delay_ms: int,
    connections: int,
    concurrency: int,
    rounds: int,
    warmup_rounds: int,
    recv_timeout_s: float,
    max_size: int,
    jsonl_out: Path | None,
) -> list[RoundMetrics]:
    pcm16 = _read_wav_pcm16(input_wav)
    sem = asyncio.Semaphore(concurrency)

    async def _one(cid: int) -> list[RoundMetrics]:
        async with sem:
            return await run_client_benchmark(
                client_id=cid,
                url=url,
                model=model,
                pcm16=pcm16,
                chunk_ms=chunk_ms,
                send_delay_ms=send_delay_ms,
                rounds=rounds,
                warmup_rounds=warmup_rounds,
                recv_timeout_s=recv_timeout_s,
                max_size=max_size,
            )

    tasks = [asyncio.create_task(_one(i + 1), name=f"rt-bench-{i + 1}") for i in range(connections)]
    parts = await asyncio.gather(*tasks)
    flat: list[RoundMetrics] = []
    for p in parts:
        flat.extend(p)

    if jsonl_out is not None:
        jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_out.open("w", encoding="utf-8") as f:
            for m in flat:
                row = asdict(m)
                # JSON-friendly
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return flat


def _print_summary(rows: list[RoundMetrics]) -> None:
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
    wall = total_e2e  # sum of per-round e2e (not global wall — document)
    if total_e2e > 0:
        print(
            f"  aggregate_tokens_per_s (sum_tokens/sum_e2e): {total_tokens / total_e2e:.4f} "
            f"(sum_completion_tokens={total_tokens}, sum_e2e_s={total_e2e:.4f})"
        )
    if total_e2e > 0 and total_audio_s > 0:
        print(f"  aggregate_rtf (sum_e2e/sum_output_audio_s): {total_e2e / total_audio_s:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark /v1/realtime WebSocket streaming audio")
    parser.add_argument("--url", default="ws://localhost:8091/v1/realtime", help="WebSocket URL")
    parser.add_argument("--model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct", help="Model for session.update")
    parser.add_argument("--input-wav", required=True, type=Path, help="Input WAV (mono, PCM16, 16 kHz)")
    parser.add_argument("--chunk-ms", type=int, default=200, help="Input chunk size in milliseconds")
    parser.add_argument("--send-delay-ms", type=int, default=0, help="Delay between chunk sends (0 = as fast as possible)")
    parser.add_argument("--connections", type=int, default=1, help="Number of parallel WebSocket clients")
    parser.add_argument("--concurrency", type=int, default=1, help="Max concurrent connections (<= connections)")
    parser.add_argument("--rounds", type=int, default=1, help="Counted rounds per connection after warmup")
    parser.add_argument("--warmup-rounds", type=int, default=0, help="Rounds per connection that are not recorded")
    parser.add_argument("--recv-timeout", type=float, default=600.0, help="Per-message recv timeout (seconds)")
    parser.add_argument(
        "--max-ws-message-mb",
        type=int,
        default=64,
        help="websockets max_size in MiB (default 64, same order as example client)",
    )
    parser.add_argument("--jsonl-out", type=Path, default=None, help="Write one JSON object per round to this path")

    args = parser.parse_args()

    if args.connections < 1:
        raise ValueError("--connections must be >= 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be >= 1")
    if args.rounds < 1:
        raise ValueError("--rounds must be >= 1")
    if args.warmup_rounds < 0:
        raise ValueError("--warmup-rounds must be >= 0")

    concurrency = min(args.concurrency, args.connections)
    max_size = args.max_ws_message_mb * 1024 * 1024

    rows = asyncio.run(
        run_benchmark(
            url=args.url,
            model=args.model,
            input_wav=args.input_wav,
            chunk_ms=args.chunk_ms,
            send_delay_ms=args.send_delay_ms,
            connections=args.connections,
            concurrency=concurrency,
            rounds=args.rounds,
            warmup_rounds=args.warmup_rounds,
            recv_timeout_s=args.recv_timeout,
            max_size=max_size,
            jsonl_out=args.jsonl_out,
        )
    )

    _print_summary(rows)
    failed_any = any(r.error for r in rows)
    if failed_any:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

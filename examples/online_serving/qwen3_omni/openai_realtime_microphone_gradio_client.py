"""Gradio realtime microphone client for vLLM-Omni /v1/realtime.

This demo is intentionally close to upstream vLLM's Gradio microphone example,
but targets audio-in/audio-out realtime flow:
1) Browser microphone streams audio chunks to /v1/realtime,
2) Server returns response.audio.delta chunks,
3) Gradio plays streamed output audio in the page.

Usage:
  python openai_realtime_microphone_gradio_client.py --host localhost --port 8091

Dependencies:
  pip install websockets numpy gradio
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import queue
import threading
import time
import traceback
import wave
from pathlib import Path

import gradio as gr
import numpy as np
import websockets

INPUT_SAMPLE_RATE = 16_000

# Global runtime state (single-session demo style, same as upstream example).
audio_input_queue: queue.Queue[str] = queue.Queue()
audio_output_queue: queue.Queue[tuple[int, np.ndarray]] = queue.Queue()
is_running = False
transcription_text = ""
status_text = "idle"
output_audio_pcm = bytearray()
output_audio_sr = 24_000
ws_url = ""
model = ""
output_wav_path: Path | None = None
debug_enabled = False

state_lock = threading.Lock()


def _dbg(msg: str) -> None:
    if not debug_enabled:
        return
    ts = time.strftime("%H:%M:%S")
    print(f"[rt-gradio {ts}] {msg}", flush=True)


def _write_wav_pcm16(path: Path, pcm16_bytes: bytes, sample_rate_hz: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate_hz)
        wf.writeframes(pcm16_bytes)


async def websocket_handler() -> None:
    global is_running, transcription_text, status_text, output_audio_sr
    final_commit_sent = False
    generation_started = False
    response_done = asyncio.Event()
    sent_chunks = 0
    recv_audio_deltas = 0
    recv_text_deltas = 0

    try:
        _dbg(f"connecting websocket: {ws_url}")
        async with websockets.connect(ws_url, max_size=64 * 1024 * 1024) as ws:
            with state_lock:
                status_text = "connected"
            _dbg("websocket connected")

            # Read the first server frame before sending client events.
            # If server rejects realtime mode (e.g., async_chunk enabled), it may send
            # an error and close immediately; reading first makes root cause visible.
            try:
                first_message = await asyncio.wait_for(ws.recv(), timeout=5.0)
                if isinstance(first_message, bytes):
                    _dbg(f"recv first binary frame bytes={len(first_message)}")
                else:
                    first_event = json.loads(first_message)
                    first_type = first_event.get("type")
                    _dbg(f"recv first event type={first_type} payload={first_event}")
                    if first_type == "error":
                        with state_lock:
                            status_text = f"server error: {first_event}"
                        return
            except asyncio.TimeoutError:
                _dbg("recv first event timeout (continue)")

            # Validate model after handshake frame.
            await ws.send(json.dumps({"type": "session.update", "model": model}))
            _dbg(f"sent session.update model={model}")

            async def send_audio() -> None:
                nonlocal final_commit_sent, generation_started, sent_chunks
                loop = asyncio.get_running_loop()
                while True:
                    try:
                        chunk_b64 = await loop.run_in_executor(
                            None, lambda: audio_input_queue.get(timeout=0.1)
                        )
                        if not generation_started:
                            # Start realtime generation only when we have first audio chunk.
                            await ws.send(
                                json.dumps({"type": "input_audio_buffer.commit", "final": False})
                            )
                            generation_started = True
                            _dbg("sent input_audio_buffer.commit(final=False)")
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "input_audio_buffer.append",
                                    "audio": chunk_b64,
                                }
                            )
                        )
                        sent_chunks += 1
                        if sent_chunks == 1 or sent_chunks % 20 == 0:
                            _dbg(f"sent append chunks={sent_chunks}")
                    except queue.Empty:
                        # When user stops recording, send final commit once.
                        if not is_running and not final_commit_sent:
                            if generation_started:
                                await ws.send(
                                    json.dumps({"type": "input_audio_buffer.commit", "final": True})
                                )
                                _dbg("sent input_audio_buffer.commit(final=True)")
                            final_commit_sent = True
                        if final_commit_sent and response_done.is_set():
                            _dbg("send_audio loop exits: final_commit_sent and response_done")
                            break
                        continue

            async def receive_events() -> None:
                global transcription_text, output_audio_sr
                nonlocal recv_audio_deltas, recv_text_deltas
                async for message in ws:
                    if isinstance(message, bytes):
                        _dbg(f"recv binary frame bytes={len(message)}")
                        continue

                    event = json.loads(message)
                    event_type = event.get("type")
                    _dbg(f"recv event type={event_type}")

                    if event_type == "session.created":
                        continue

                    if event_type == "transcription.delta":
                        delta = event.get("delta", "")
                        if delta:
                            recv_text_deltas += 1
                            with state_lock:
                                transcription_text += delta
                            if recv_text_deltas == 1 or recv_text_deltas % 10 == 0:
                                _dbg(f"recv transcription.delta count={recv_text_deltas}")
                        continue

                    if event_type == "transcription.done":
                        final_text = event.get("text", "")
                        if final_text:
                            with state_lock:
                                transcription_text = final_text
                        _dbg("recv transcription.done")
                        continue

                    if event_type == "response.audio.delta":
                        recv_audio_deltas += 1
                        sr = event.get("sample_rate_hz")
                        if isinstance(sr, int) and sr > 0:
                            output_audio_sr = sr
                        audio_b64 = event.get("audio", "")
                        if not audio_b64:
                            continue
                        pcm_bytes = base64.b64decode(audio_b64)
                        pcm_np = np.frombuffer(pcm_bytes, dtype=np.int16).copy()
                        if pcm_np.size == 0:
                            continue
                        with state_lock:
                            output_audio_pcm.extend(pcm_bytes)
                        audio_output_queue.put((output_audio_sr, pcm_np))
                        if recv_audio_deltas == 1 or recv_audio_deltas % 10 == 0:
                            _dbg(
                                f"recv response.audio.delta count={recv_audio_deltas}, "
                                f"pcm_bytes={len(pcm_bytes)}, sr={output_audio_sr}"
                            )
                        continue

                    if event_type == "response.audio.done":
                        _dbg("recv response.audio.done")
                        response_done.set()
                        break

                    if event_type == "error":
                        with state_lock:
                            status_text = f"server error: {event}"
                        _dbg(f"recv error event: {event}")
                        response_done.set()
                        break

                _dbg("receive_events loop ended")
            await asyncio.gather(send_audio(), receive_events())
            _dbg(
                f"websocket_handler done: sent_chunks={sent_chunks}, "
                f"recv_audio_deltas={recv_audio_deltas}, recv_text_deltas={recv_text_deltas}"
            )

    except Exception as exc:
        with state_lock:
            status_text = f"websocket error: {exc}"
        _dbg(f"websocket exception: {exc}")
        _dbg(traceback.format_exc())
    finally:
        with state_lock:
            is_running = False
            if status_text == "connected":
                status_text = "finished"
            elif status_text == "starting websocket...":
                status_text = "stopped"
        _dbg("websocket handler finally: set is_running=False")

        if output_wav_path is not None:
            with state_lock:
                pcm_bytes = bytes(output_audio_pcm)
                sr = output_audio_sr
            if pcm_bytes:
                _write_wav_pcm16(output_wav_path, pcm_bytes, sr)
                with state_lock:
                    status_text = f"finished, saved output to {output_wav_path}"
                _dbg(f"saved wav: {output_wav_path}, bytes={len(pcm_bytes)}, sr={sr}")
            else:
                _dbg("no output audio saved (empty pcm)")


def start_websocket_thread() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(websocket_handler())


def start_recording():
    global is_running, transcription_text, status_text, output_audio_pcm, output_audio_sr
    with state_lock:
        transcription_text = ""
        status_text = "starting websocket..."
        output_audio_pcm = bytearray()
        output_audio_sr = 24_000
        is_running = True

    while not audio_input_queue.empty():
        try:
            audio_input_queue.get_nowait()
        except queue.Empty:
            break
    while not audio_output_queue.empty():
        try:
            audio_output_queue.get_nowait()
        except queue.Empty:
            break

    thread = threading.Thread(target=start_websocket_thread, daemon=True)
    thread.start()
    _dbg("start button clicked: websocket thread started")
    return (
        gr.update(interactive=False),
        gr.update(interactive=True),
        "",
        None,
        "running",
    )


def stop_recording():
    global is_running
    with state_lock:
        is_running = False
    _dbg("stop button clicked: set is_running=False")
    return (
        gr.update(interactive=True),
        gr.update(interactive=False),
        gr.update(),
        gr.update(),
        "stopping...",
    )


def process_audio(audio) -> None:
    """Consume gradio microphone chunks and enqueue base64 PCM16 for websocket."""
    if audio is None or not is_running:
        return

    sample_rate, audio_data = audio
    if audio_data is None:
        return

    # Convert to mono if browser provides stereo.
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    if audio_data.dtype == np.int16:
        audio_f32 = audio_data.astype(np.float32) / 32767.0
    else:
        audio_f32 = audio_data.astype(np.float32)

    # Resample to 16k for realtime input requirement.
    if sample_rate != INPUT_SAMPLE_RATE:
        num_samples = int(len(audio_f32) * INPUT_SAMPLE_RATE / sample_rate)
        if num_samples <= 0:
            return
        audio_f32 = np.interp(
            np.linspace(0, len(audio_f32) - 1, num_samples),
            np.arange(len(audio_f32)),
            audio_f32,
        )

    pcm16 = np.clip(audio_f32, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    b64_chunk = base64.b64encode(pcm16.tobytes()).decode("utf-8")
    audio_input_queue.put(b64_chunk)
    if debug_enabled and audio_input_queue.qsize() == 1:
        _dbg(
            f"first mic chunk queued: input_sr={sample_rate}, "
            f"resampled_to={INPUT_SAMPLE_RATE}, samples={len(pcm16)}"
        )


def poll_updates():
    """Periodic UI updater for transcription text and streamed output audio."""
    with state_lock:
        text = transcription_text
        status = status_text

    audio_chunk = None
    latest_sr = None
    chunks: list[np.ndarray] = []
    while not audio_output_queue.empty():
        try:
            sr, pcm = audio_output_queue.get_nowait()
            latest_sr = sr
            chunks.append(pcm)
        except queue.Empty:
            break
    if chunks and latest_sr is not None:
        merged = np.concatenate(chunks)
        # Gradio Audio accepts (sample_rate, np.ndarray).
        audio_chunk = (latest_sr, merged)

    return text, audio_chunk, status


with gr.Blocks(title="vLLM-Omni Realtime Microphone Audio Demo") as demo:
    gr.Markdown("# vLLM-Omni Realtime Audio Chat (Microphone)")
    gr.Markdown(
        "Click **Start**, speak into your microphone, and hear streamed model audio output."
    )

    with gr.Row():
        start_btn = gr.Button("Start", variant="primary")
        stop_btn = gr.Button("Stop", variant="stop", interactive=False)

    audio_input = gr.Audio(
        sources=["microphone"],
        streaming=True,
        type="numpy",
        label="Microphone Input",
    )
    audio_output = gr.Audio(
        streaming=True,
        autoplay=True,
        label="Model Audio Output",
    )
    transcription_output = gr.Textbox(label="Transcription", lines=6)
    status_output = gr.Textbox(label="Status")
    poll_timer = gr.Timer(0.15)

    start_btn.click(
        start_recording,
        outputs=[start_btn, stop_btn, transcription_output, audio_output, status_output],
    )
    stop_btn.click(
        stop_recording,
        outputs=[start_btn, stop_btn, transcription_output, audio_output, status_output],
    )
    audio_input.stream(process_audio, inputs=[audio_input], outputs=[])
    poll_timer.tick(
        poll_updates,
        outputs=[transcription_output, audio_output, status_output],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gradio realtime microphone audio-in/audio-out demo for vLLM-Omni"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed websocket and audio flow logs for troubleshooting.",
    )
    parser.add_argument(
        "--output-wav",
        type=Path,
        default=Path("realtime_gradio_output.wav"),
        help="Where to save concatenated streamed output audio.",
    )
    args = parser.parse_args()

    ws_url = f"ws://{args.host}:{args.port}/v1/realtime"
    model = args.model
    output_wav_path = args.output_wav
    debug_enabled = args.debug

    print(
        "[rt-gradio] script=openai_realtime_microphone_gradio_client.py "
        f"debug={debug_enabled} ws_url={ws_url} model={model}",
        flush=True,
    )

    demo.launch(share=args.share)

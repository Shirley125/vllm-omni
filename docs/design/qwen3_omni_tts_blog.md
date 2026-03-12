# Speech Generation on vLLM-Omni: Performance Optimizations for Qwen3-Omni and Qwen3-TTS

## Summary

vLLM-Omni supports end-to-end serving for speech-generating models, including both **Qwen3-Omni** (multimodal understanding + speech) and **Qwen3-TTS** (text-to-speech). Despite their different architectures, both models share the same multi-stage pipeline design and benefit from the same set of stacked optimizations:

1. **Batching** improves GPU utilization stage by stage and increases overall throughput.
2. **CUDA Graph** reduces CPU launch overhead and decode-time jitter on stable shapes.
3. **Async Chunk and Streaming Output** overlap compute and communication across stages and emit audio incrementally, improving both TTFP and E2E.

### Model architectures

**Qwen3-Omni** is a native multimodal model that understands text, audio, image, and video inputs, and generates both text and speech outputs. Its pipeline has three stages:

- **Thinker**: multimodal understanding and text generation
- **Talker (+ Talker-MTP / code predictor path)**: converts semantic/text representations into codec tokens
- **Code2Wav**: decodes codec tokens into waveform audio

**Qwen3-TTS** is a lightweight, high-quality text-to-speech model. Its pipeline has two stages:

- **Talker (AR decoder)**: auto-regressively generates codec tokens from text input
- **Code2Wav (vocoder)**: decodes codec tokens into waveform audio

The optimizations described in this post apply to both models. We present results for each side by side.

### vLLM-Omni vs HF Transformers

Compared with **HF Transformers** (offline, single request), vLLM-Omni with the full optimization stack delivers dramatically lower latency and higher efficiency for both models.

**Qwen3-Omni** (A100):

<table><tr>
<td><img src="figures/omni/E2EL_s_vllm_omni_vs_transformers.png" alt="Omni E2EL: vLLM-Omni vs HF" width="100%"/></td>
<td><img src="figures/omni/TTFP_s_vllm_omni_vs_transformers.png" alt="Omni TTFP: vLLM-Omni vs HF" width="100%"/></td>
<td><img src="figures/omni/RTF_vllm_omni_vs_transformers.png" alt="Omni RTF: vLLM-Omni vs HF" width="100%"/></td>
</tr></table>

- **E2E latency**: 23.78 s vs 336.10 s - **~93%** reduction
- **TTFP**: 0.934 s vs 336.10 s - **~99.7%** reduction
- **RTF**: 0.32 vs 3.776 - **~91%** reduction (~12x faster)

**Qwen3-TTS** (H200, concurrency 1):

<table><tr>
<td><img src="figures/tts/Mean_E2EL_(ms)_vllm_omni_vs_transformers.png" alt="TTS E2EL: vLLM-Omni vs HF" width="100%"/></td>
<td><img src="figures/tts/Mean_AUDIO_TTFP_(ms)_vllm_omni_vs_transformers.png" alt="TTS TTFP: vLLM-Omni vs HF" width="100%"/></td>
<td><img src="figures/tts/Mean_AUDIO_RTF_vllm_omni_vs_transformers.png" alt="TTS RTF: vLLM-Omni vs HF" width="100%"/></td>
</tr></table>

- **E2E latency**: 2.08 s vs 15.51 s - **~87%** reduction
- **TTFP**: 97 ms vs 15,513 ms - **~99.4%** reduction (160x faster)
- **RTF**: 0.34 vs 2.64 - **~87%** reduction (~7.8x faster)

### Stacked optimization summary

Each optimization stacks on the previous one. The summary plots below show the cumulative effect at each step, with one line per concurrency level (1, 4, 10).

**Qwen3-Omni** (A100):

<table><tr>
<td><img src="figures/omni/Summary_E2EL_ms_vs_features.png" alt="Omni E2EL summary" width="100%"/></td>
<td><img src="figures/omni/Summary_TTFP_ms_vs_features.png" alt="Omni TTFP summary" width="100%"/></td>
<td><img src="figures/omni/Summary_RTF_vs_features.png" alt="Omni RTF summary" width="100%"/></td>
</tr></table>

- **E2EL reduction**: ~91% at concurrency 10 (1,523,135 ms -> 130,682 ms); ~81% at concurrency 1 (325,865 ms -> 60,436 ms)
- **TTFP reduction**: ~99.2% at concurrency 10 (1,522,804 ms -> 12,262 ms); ~99.6% at concurrency 1 (325,517 ms -> 1,263 ms)
- **RTF reduction**: ~89% at concurrency 10 (6.94 -> 0.74); ~78% at concurrency 1 (1.52 -> 0.33)

**Qwen3-TTS** (H200):

<table><tr>
<td><img src="figures/tts/Summary_mean_e2e_ms_vs_features.png" alt="TTS E2EL summary" width="100%"/></td>
<td><img src="figures/tts/Summary_mean_ttfp_ms_vs_features.png" alt="TTS TTFP summary" width="100%"/></td>
<td><img src="figures/tts/Summary_mean_rtf_vs_features.png" alt="TTS RTF summary" width="100%"/></td>
</tr></table>

- **E2EL reduction**: ~89% at concurrency 10 (44,543 ms -> 4,971 ms); ~56% at concurrency 1 (4,714 ms -> 2,082 ms)
- **TTFP reduction**: ~97% at concurrency 10 (44,543 ms -> 1,383 ms); ~98% at concurrency 1 (4,714 ms -> 97 ms)
- **RTF reduction**: ~89% at concurrency 10 (8.15 -> 0.90); ~60% at concurrency 1 (0.85 -> 0.34)

**Benchmark environment:**

| | Qwen3-Omni | Qwen3-TTS |
| --- | --- | --- |
| **GPU** | A100 | H200 |
| **Model** | Qwen3-Omni-30B-A3B-Instruct | Qwen3-TTS-12Hz-1.7B-CustomVoice |
| **vLLM** | v0.16.0 | v0.16.0 |
| **vllm-omni** | commit 070ea0dd | commit TODO |
| **CUDA** | 12.8 | 12.8 |

This post walks through each optimization in the same order they are typically enabled in practice, then ends with deployment playbooks for both models.

---

## Pipeline Batching

### How stage-wise batching works

For both Qwen3-Omni and Qwen3-TTS, batching is a pipeline-level optimization:

- Requests are grouped per stage using `runtime.max_batch_size`
- Each stage executes batch inference with its own scheduler/worker
- Stage outputs are routed to downstream stages with per-request mapping preserved

**Batching strategy by stage:** The understanding and decode stages (Thinker for Omni, Talker for both) use **continuous batching**: requests can join and leave the batch over time. Code2Wav uses **static batching**: once a batch is formed, the stage runs the whole batch before starting the next. This matches the decode pattern of Code2Wav and keeps implementation simple while still improving throughput.

### Batching results (Baseline vs. Batch)

Batching alone greatly reduces E2EL and RTF across all concurrencies. The biggest gains appear at high concurrency where requests share GPU resources.

**Qwen3-Omni** (A100):

<table><tr>
<td><img src="figures/omni/Mean_E2EL_ms_Baseline_vs_Batch.png" alt="Omni E2EL: Baseline vs Batch" width="100%"/></td>
<td><img src="figures/omni/Mean_AUDIO_TTFP_ms_Baseline_vs_Batch.png" alt="Omni TTFP: Baseline vs Batch" width="100%"/></td>
<td><img src="figures/omni/Mean_AUDIO_RTF_Baseline_vs_Batch.png" alt="Omni RTF: Baseline vs Batch" width="100%"/></td>
</tr></table>

At concurrency 10, E2EL drops from ~1,523 s to ~262 s; at concurrency 1, from ~326 s to ~239 s.

**Qwen3-TTS** (H200):

<table><tr>
<td><img src="figures/tts/Mean_mean_e2e_ms_baseline_vs_batch.png" alt="TTS E2EL: Baseline vs Batch" width="100%"/></td>
<td><img src="figures/tts/Mean_mean_ttfp_ms_baseline_vs_batch.png" alt="TTS TTFP: Baseline vs Batch" width="100%"/></td>
<td><img src="figures/tts/Mean_mean_rtf_baseline_vs_batch.png" alt="TTS RTF: Baseline vs Batch" width="100%"/></td>
</tr></table>

| Metric | Concurrency | Baseline | + Batch | Improvement |
| --- | --- | --- | --- | --- |
| E2EL (ms) | 1 | 4,714 | 2,531 | 1.9x |
| E2EL (ms) | 4 | 18,258 | 2,847 | 6.4x |
| E2EL (ms) | 10 | 44,543 | 4,920 | 9.1x |
| RTF | 1 | 0.851 | 0.434 | 2.0x |
| RTF | 4 | 3.205 | 0.492 | 6.5x |
| RTF | 10 | 8.152 | 0.871 | 9.4x |
| Throughput | 10 | 1.25x | 10.35x | 8.3x |

At concurrency 10, batching alone brings Qwen3-TTS RTF from 8.15 (far from realtime) down to 0.87 (faster than realtime), and throughput from 1.25x to 10.35x.

---

## CUDA Graph on the Critical Decode Path

### Why CUDA Graph helps here

In decode-heavy serving, repeatedly launching many small kernels from CPU can become a visible overhead. CUDA Graph reduces this overhead by capturing and replaying stable execution graphs.

In stage configs, this is represented by `enforce_eager: false` for stages where graph capture is desired (Thinker/Talker), while Code2Wav keeps eager mode depending on stage behavior.

### CUDA Graph results on top of batching

**Qwen3-Omni** (A100):

<table><tr>
<td><img src="figures/omni/Mean_E2EL_ms_Batch_vs_Batch_CUDA_Graph.png" alt="Omni E2EL: Batch vs +CG" width="100%"/></td>
<td><img src="figures/omni/Mean_AUDIO_TTFP_ms_Batch_vs_Batch_CUDA_Graph.png" alt="Omni TTFP: Batch vs +CG" width="100%"/></td>
<td><img src="figures/omni/Mean_AUDIO_RTF_Batch_vs_Batch_CUDA_Graph.png" alt="Omni RTF: Batch vs +CG" width="100%"/></td>
</tr></table>

For the larger Qwen3-Omni model (30B-A3B), CUDA Graph provides a significant improvement. At concurrency 1, E2EL drops from ~239 s to ~67 s; at concurrency 10, from ~262 s to ~153 s.

**Qwen3-TTS** (H200):

For the Qwen3-TTS 1.7B model, vLLM's CUDA graph on the main Talker decode loop shows **negligible impact** - within noise at concurrency 1 and slightly worse at concurrency 10. The model is small enough that CPU kernel launch overhead is already a tiny fraction of total compute, so graph replay doesn't help. For larger TTS models, the benefit would be more pronounced.

Instead, the TTS-specific kernel-level optimization comes from **`torch.compile` on the code predictor** (see below), which targets the many small kernels in the 5-layer residual codebook transformer rather than the main Talker decode path.

---

## Async Chunk and Streaming Output: Earlier Audio and Cross-Stage Overlap

### Why this step matters for first-packet latency

Two mechanisms work together to improve user-visible latency:

- **Streaming output**: audio streaming emits audio chunks as soon as they are decoded (lower **TTFP**). Without streaming, the client waits for larger buffers or end-of-sequence.
- **Async chunk** is the main enabler for *earlier* audio: instead of handing off whole-request results between stages, each stage forwards **chunks** so the next stage can start as soon as the first chunk is ready. For Omni: Thinker -> Talker forwards hidden-state chunks; for both: Talker -> Code2Wav forwards codec chunks; Code2Wav decodes and emits packets incrementally. This **overlaps compute and communication** across stages and directly reduces time-to-first-audio-packet (TTFP) and end-to-end latency (E2EL).

So in practice: streaming output defines *how* bytes are sent to the client; async chunk defines *when* the pipeline can produce the first bytes.

**Dependency between the two:** Async chunk and audio streaming output are mutually dependent. Without async chunk, **audio streaming output cannot truly take effect**. Without audio streaming output, async chunk's **TTFP advantage is not fully realized**: the client would still wait for larger buffers or end-of-sequence instead of hearing the first packet as soon as it is ready. We therefore recommend enabling **both** on top of batching + CUDA Graph; the benchmarks in this post use both.

### Results: Batch + CUDA Graph vs. Batch + CUDA Graph + Async Chunk + Streaming Output

**Qwen3-Omni** (A100):

<table><tr>
<td><img src="figures/omni/Mean_E2EL_ms_Batch_CUDA_Graph_vs_Async_Chunk.png" alt="Omni E2EL: +CG vs +Async" width="100%"/></td>
<td><img src="figures/omni/Mean_AUDIO_TTFP_ms_Batch_CUDA_Graph_vs_Async_Chunk.png" alt="Omni TTFP: +CG vs +Async" width="100%"/></td>
<td><img src="figures/omni/Mean_AUDIO_RTF_Batch_CUDA_Graph_vs_Async_Chunk.png" alt="Omni RTF: +CG vs +Async" width="100%"/></td>
</tr></table>

Enabling both brings TTFP down sharply (concurrency 1: 67,121 ms -> 1,263 ms, **~98% reduction**; concurrency 4: 98,679 ms -> 3,175 ms, **~97% reduction**). E2EL and RTF also improve at every concurrency.

**Qwen3-TTS** (H200):

<table><tr>
<td><img src="figures/tts/Mean_mean_e2e_ms_cuda_graph_vs_async_chunk.png" alt="TTS E2EL: +CG vs +Async" width="100%"/></td>
<td><img src="figures/tts/Mean_mean_ttfp_ms_cuda_graph_vs_async_chunk.png" alt="TTS TTFP: +CG vs +Async" width="100%"/></td>
<td><img src="figures/tts/Mean_mean_rtf_cuda_graph_vs_async_chunk.png" alt="TTS RTF: +CG vs +Async" width="100%"/></td>
</tr></table>

| Metric | Concurrency | Batch + CG | + Async Chunk | Improvement |
| --- | --- | --- | --- | --- |
| TTFP (ms) | 1 | 2,335 | **97** | **24.1x** |
| TTFP (ms) | 4 | 3,975 | **254** | **15.7x** |
| TTFP (ms) | 10 | 7,743 | **1,383** | **5.6x** |
| E2EL (ms) | 1 | 2,335 | 2,082 | 1.1x |
| E2EL (ms) | 10 | 7,743 | 4,971 | 1.6x |
| RTF | 1 | 0.407 | 0.340 | 1.2x |
| RTF | 10 | 1.403 | 0.898 | 1.6x |

The TTFP improvement is the headline result for both models. For Qwen3-TTS at concurrency 1, users hear the first audio in **97 ms** instead of 2,335 ms - a **24x reduction**. For Qwen3-Omni at concurrency 1, TTFP drops from 67 s to 1.3 s - a **53x reduction**.

---

## TTS-Specific: Code Predictor Re-prefill + `torch.compile`

Qwen3-TTS has a **code predictor** - a small 5-layer transformer that generates residual codebook tokens (groups 1 through Q-1) autoregressively. Each AR step operates on very short sequences (2 to ~16 tokens).

The naive approach uses a KV cache for this small transformer, similar to the main Talker. But the KV cache machinery (block tables, slot mappings, paged attention) introduces significant overhead relative to the tiny model. Two optimizations replace that:

### Re-prefill (stateless forward, no KV cache)

Instead of maintaining a KV cache across steps, the code predictor **re-feeds the full growing sequence** at each AR step using `F.scaled_dot_product_attention`. With sequences of at most ~16 tokens through 5 layers, the O(T^2) attention cost is negligible - and removing the KV cache machinery (block table management, `set_forward_context`, slot mapping) saves far more time than it costs.

### `torch.compile` on the code predictor forward

The 5-layer transformer forward pass launches ~60 small CUDA kernels per step. `torch.compile(mode="default", dynamic=True)` fuses these into fewer kernels via Inductor:

```python
self._compiled_model_fwd = torch.compile(
    self.model.forward,
    mode="default",    # no Inductor CUDA graphs, avoids conflict with vLLM's CUDAGraphWrapper
    dynamic=True,      # sequence length grows each step (2, 3, ..., num_groups+1)
)
```

`mode="default"` is used instead of `mode="reduce-overhead"` to avoid conflicts with vLLM's own CUDA graph capture on the main Talker model. `dynamic=True` handles the growing sequence length without recompilation.

These optimizations are always-on in the current codebase - all Qwen3-TTS benchmark results in this post include them.

---

## TTS-Specific: Dynamic Initial Chunk for Faster First Audio

In the async chunk pipeline, the standard `codec_chunk_frames` is 25 (each chunk = ~2 seconds of audio at 12 Hz). Waiting for 25 frames before forwarding the first chunk to Code2Wav adds unnecessary TTFP. The **initial codec chunk** optimization sends a smaller first chunk so Code2Wav can start decoding earlier.

**Dynamic initial chunk sizing (default behavior):**

Rather than using a fixed initial chunk size, vLLM-Omni dynamically selects it based on current server load. The initial chunk size is chosen from power-of-2 steps [2, 4, 8, 16] based on load factor (`active_requests / max_batch_size`):

| Server load | Initial chunk frames | Rationale |
| --- | --- | --- |
| Low (e.g. 1/10 active) | **2** (~167 ms of audio) | Minimize TTFP when there's headroom |
| Medium (e.g. 5/10 active) | **4-8** | Balance TTFP vs decode efficiency |
| High (e.g. 10/10 active) | **16** | Larger first chunk to amortize decode cost |

After the initial chunk, all subsequent chunks use the standard `codec_chunk_frames` (25) size.

**How it works in the pipeline:**

1. Talker generates codec tokens auto-regressively
2. The stage input processor checks current load and picks an initial chunk size (e.g. **2 frames** at low load)
3. After that many frames, the first chunk is forwarded to Code2Wav
4. Code2Wav decodes this small chunk and emits the first audio packet
5. Subsequent chunks use the standard 25-frame size for efficient batch decoding

**Per-request override:** Clients can also set a fixed initial chunk size via the API:

```json
{"initial_codec_chunk_frames": 2}
```

This overrides the dynamic calculation for that request.

**Config (server-side):**

```yaml
runtime:
  connectors:
    connector_of_shared_memory:
      name: SharedMemoryConnector
      extra:
        codec_streaming: true
        codec_chunk_frames: 25              # standard chunk size (~2s of audio)
        codec_left_context_frames: 25
        # initial chunk is computed dynamically by default
        # set initial_codec_chunk_frames: 2 to force a fixed value
```

The 97 ms TTFP result reported above for Qwen3-TTS at concurrency 1 uses the dynamic initial chunk, which picks `initial_codec_chunk_frames=2` at low load. At higher concurrency the dynamic sizing increases the initial chunk to maintain decode efficiency.

---

## Live Demo: Streaming TTS over WebSocket

vLLM-Omni supports real-time streaming audio output for Qwen3-TTS over WebSocket ([PR #1719](https://github.com/vllm-project/vllm-omni/pull/1719)). With `stream_audio: true`, the server sends chunked PCM audio frames as they are generated, so clients can start playback before full sentence synthesis completes.

The WebSocket protocol uses `audio.start` / binary PCM chunks / `audio.done` framing per sentence:

```json
// Client sends:
{"type":"session.config","voice":"Vivian","response_format":"pcm","stream_audio":true}
{"type":"input.text","text":"Hello world. This is a streaming demo."}
{"type":"input.done"}

// Server streams back per sentence:
{"type":"audio.start","sentence_index":0,"sentence_text":"Hello world.","format":"pcm","sample_rate":24000}
<binary PCM chunk 1>
<binary PCM chunk 2>
...
{"type":"audio.done","sentence_index":0,"total_bytes":96000,"error":false}
{"type":"audio.start","sentence_index":1,"sentence_text":"This is a streaming demo.","format":"pcm","sample_rate":24000}
<binary PCM chunk 1>
...
{"type":"audio.done","sentence_index":1,"total_bytes":72000,"error":false}
{"type":"session.done","total_sentences":2}
```

<iframe width="560" height="315" src="https://www.youtube.com/embed/rB_MgPUx46U" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---

## Deployment Playbook

### Qwen3-Omni

#### 1) Serve with the default 3-stage config

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni \
  --port 8091
```

Notes:

- `runtime.max_batch_size` controls stage-level batching.
- Thinker/Talker commonly use `enforce_eager: false` for CUDA Graph paths.
- Code2Wav often remains eager (`enforce_eager: true`) depending on runtime behavior.

#### 2) Enable async chunk

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni \
  --port 8091 \
  --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_omni_moe_async_chunk.yaml
```

#### 3) Key config knobs

```yaml
async_chunk: true
stage_args:
  - stage_id: 0  # thinker
    runtime:
      max_batch_size: 64
    engine_args:
      enforce_eager: false
      max_num_batched_tokens: 32768
      custom_process_next_stage_input_func: >-
        vllm_omni.model_executor.stage_input_processors.qwen3_omni.thinker2talker_async_chunk

  - stage_id: 1  # talker
    runtime:
      max_batch_size: 64
    engine_args:
      enforce_eager: false
      max_num_batched_tokens: 32768
      custom_process_next_stage_input_func: >-
        vllm_omni.model_executor.stage_input_processors.qwen3_omni.talker2code2wav_async_chunk

  - stage_id: 2  # code2wav
    runtime:
      max_batch_size: 64
    engine_args:
      enforce_eager: true
      max_num_batched_tokens: 51200
```

#### Reproduce Qwen3-Omni benchmarks

```bash
vllm bench serve \
  --dataset-name random \
  --port ${PORT} \
  --model ${MODEL_PATH} \
  --endpoint /v1/chat/completions \
  --backend openai-chat-omni \
  --max-concurrency ${MAX_CONCURRENCY} \
  --num-prompts ${NUM_PROMPTS} \
  --random-input-len 2500 \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el,audio_ttfp,audio_rtf \
  --random-output-len 900 \
  --extra_body '{"modalities": ["text","audio"]}'
```

### Qwen3-TTS

#### 1) Serve with async chunk (recommended)

```bash
vllm-omni serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --omni \
  --port 8000
```

The default config (`qwen3_tts.yaml`) enables the full optimization stack:

- Batching with `max_batch_size: 10` on the Talker stage
- CUDA Graph on the Talker (`enforce_eager: false`)
- Async chunk with streaming transport

#### 2) Serve without async chunk (for comparison)

```bash
vllm-omni serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --omni \
  --port 8000 \
  --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_tts_no_async_chunk.yaml
```

#### 3) Key config knobs

```yaml
async_chunk: true
stage_args:
  - stage_id: 0  # Talker (AR decoder)
    runtime:
      max_batch_size: 10
    engine_args:
      enforce_eager: false
      max_num_batched_tokens: 512
      custom_process_next_stage_input_func: >-
        vllm_omni.model_executor.stage_input_processors.qwen3_tts.talker2code2wav_async_chunk

  - stage_id: 1  # Code2Wav (vocoder)
    runtime:
      max_batch_size: 1
    engine_args:
      enforce_eager: true
      max_num_batched_tokens: 8192

runtime:
  connectors:
    connector_of_shared_memory:
      name: SharedMemoryConnector
      extra:
        codec_streaming: true
        codec_chunk_frames: 25
        codec_left_context_frames: 25
```

#### Reproduce Qwen3-TTS benchmarks

```bash
GPU_DEVICE=0 \
MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
NUM_PROMPTS=50 \
CONCURRENCY="1 4 10" \
bash benchmarks/qwen3-tts/vllm_omni/run_stacked_benchmark.sh
```

This cycles through four configs (Baseline -> + Batch -> + CUDA Graph -> + Async Chunk + Streaming), benchmarks each at the specified concurrency levels, and generates all comparison figures automatically.

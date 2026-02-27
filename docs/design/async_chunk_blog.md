# Async Chunk: Making Multi-Stage Speech Generation Truly Streamable

In multi-stage multimodal systems like Qwen3-Omni, latency is shaped by the whole pipeline, not one model kernel.  
Async Chunk changes the serving pattern from "finish everything, then hand off" to "ship useful partial outputs immediately."

---

## Motivation: Why Async Chunk in Multi-Stage Multimodal Systems?

### 1) Full-buffer handoff creates long silence, amplifies first-packet latency

A typical speech path is **Thinker -> Talker -> Code2Wav**.  
When each stage waits for full upstream completion, first audio is delayed by stacked waiting time.

### 2) Model streaming ability is wasted without system streaming

The models like Qwen3-Omni can start producing useful outputs early, but if serving still buffers everything, that early work is invisible to users.

### 3) Concurrency amplifies queueing and jitter

At higher load, requests waiting on full upstream outputs occupy scheduler attention and hurt batch stability for everyone else.

#### Where It Matters Most
- Real-time voice assistants that care about "time to first sound"
- Multimodal applications that need fast spoken responses
- Online serving environments where first-packet stability is critical under concurrency

---

## What Changes: From Stage-Level Buffering to Chunk-Level Streaming (Principle Diagram)

Async Chunk is easiest to understand by comparing **data flow**.

### Sequential (non-async) flow: full-buffer handoff

In the sequential pattern, each stage largely waits for upstream to finish, so compute across stages cannot overlap much.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/qwen3-omni-non-async-chunk.png">
    <img alt="Sequential data flow between stages" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/qwen3-omni-non-async-chunk.png" width=100%>
  </picture>
</p>

### Async Chunk flow: forward early + overlap compute

With async chunking, upstream stage outputs are forwarded **as soon as chunks become available**, so downstream stages can begin prefill/decode earlier and run concurrently.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/qwen3-omni-async-chunk.png">
    <img alt="Async chunk data flow between stages" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/qwen3-omni-async-chunk.png" width=100%>
  </picture>
</p>

**Optimization points readers should notice from the diagrams:**
- **Earlier downstream start**: next stage can begin work right after the first chunk arrives (reduces time-to-first-audio).
- **Stage overlap**: Thinker/Talker/Code2Wav can be active at the same time for a request (improves end-to-end latency and utilization).
- **Less scheduler disruption under load**: requests that are missing chunks are explicitly parked, so runnable work keeps flowing (reduces jitter).

---

## Solution: Async Chunk Pipeline

We split the problem into three coordinated behaviors:

### 1) Chunk-level forwarding

Send partial upstream payloads as soon as they are available, instead of waiting for stage completion.

### 2) Non-blocking scheduling

Requests with missing upstream chunks move to `WAITING_FOR_CHUNK`; runnable requests continue to execute.

### 3) Controlled aggregation

For downstream speech generation (especially codec/audio stages), we aggregate where needed to avoid too many tiny kernels.

---

## Design: Key Components
Qwen3-Omni as an Example

The design separates **transport**, **chunk lifecycle**, and **scheduling hooks**.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/async-chunk-architecture.png">
    <img alt="Async Chunk Architecture" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/architecture/async-chunk-architecture.png" width=100%>
  </picture>
</p>

### OmniConnector (transport-only)

The connector only moves bytes/tensors across stages (e.g., shared memory IPC). It exposes a transport-only API:
- `put(from_stage, to_stage, put_key, data)`
- `get(from_stage, to_stage, get_key)` (optionally with timeout)

It intentionally does **not** track request lifecycle state (e.g., put/get requests, finished sets). That logic lives above.

### Chunk Transfer Adapter

`OmniChunkTransferAdapter` owns the chunk lifecycle when async_chunk is enabled:
- **Chunk key construction**: builds keys like `{req_id}_{stage_id}_{chunk_id}`
- **Async get (`load_async`)**: scheduler enqueues a load; background `recv_loop` polls connector (non-blocking); once data is ready the adapter marks the request as finished-load, and scheduler later fetches it via `get_finished_requests()`
- **Async put (`save_async`)**: main thread runs `custom_process_next_stage_input_func` to build payload / do chunk accumulation, then a background `save_loop` performs `connector.put()`

### Stage Schedulers

Before each scheduling step, schedulers call adapter hooks to:
- move chunk-waiting requests out of active queues,
- restore requests that now have data ready,
- keep queues healthy under concurrency.

### Stage Input Processors

Define how each stage converts upstream chunk payloads:
- Thinker -> Talker: embeddings/hidden states and token context
- Talker -> Code2Wav: codec code accumulation and chunked emission

---

## Request Lifecycle

Qwen3-Omni as an Example
1. A request enters Thinker.  
2. Thinker emits partial results; adapter sends chunk `0`, `1`, `2`... asynchronously.  
3. Talker polls adapter; if no chunk is ready, request stays in `WAITING_FOR_CHUNK`.  
4. Once enough chunk data is ready, Talker prefill starts and progresses chunk by chunk.  
5. Talker outputs codec chunks; Code2Wav consumes aggregated windows and emits audio packets.  
6. The final short tail is flushed when upstream marks `finished`.

The user hears audio earlier because downstream work starts before upstream is fully done.

---

## Performance Results & User Impact

Performance data below is collected on **H800 GPUs** with **cudagraph enabled**; the text input uses a random dataset.

### Benchmark table (text 100 → text 100 + audio)

| Input     | Output           | Async_chunk enabled | Code2Wav batch size | Max_Concurrency | Prompts | Mean E2E   | Mean TTFT  | Mean TPOT | Mean TTFP   | Mean RTF | Mean ITL |
|-----------|------------------|---------------------|---------------------|----------------|---------|------------|------------|-----------|-------------|----------|----------|
| text 100  | text 100+audio   | False               | 1                   | 1              | 50      | 6581.80    | 43.22      | 8.31      | 6459.34     | 0.24     | 8.22     |
| text 100  | text 100+audio   | False               | 1                   | 4              | 50      | 7398.63    | 67.57      | 9.14      | 7285.35     | 0.27     | 9.05     |
| text 100  | text 100+audio   | False               | 1                   | 10             | 50      | 13522.99   | 131.82     | 12.72     | 13410.44    | 0.49     | 12.60    |
| text 100  | text 100+audio   | False               | 64                  | 1              | 50      | 6505.13    | 43.14      | 8.52      | 6395.40     | 0.24     | 8.44     |
| text 100  | text 100+audio   | False               | 64                  | 4              | 50      | 7668.15    | 51.15      | 9.36      | 7562.37     | 0.28     | 9.27     |
| text 100  | text 100+audio   | False               | 64                  | 10             | 50      | 9516.18    | 138.06     | 14.75     | 9409.26     | 0.34     | 14.60    |
| text 100  | text 100+audio   | True                | 1                   | 1              | 50      | 6179.79    | 44.58      | 8.69      | 522.99      | 0.22     | 8.60     |
| text 100  | text 100+audio   | True                | 1                   | 4              | 50      | 7692.69    | 103.96     | 10.22     | 785.85      | 0.29     | 10.12    |
| text 100  | text 100+audio   | True                | 1                   | 10             | 50      | 11152.71   | 685.60     | 17.64     | 1628.88     | 0.41     | 17.62    |

### Key takeaways (directly reflected by the measurements)

- **Async_chunk (False→True) sharply reduces TTFP (time-to-first-audio)**  
  Example: concurrency 1, **6.5s → 0.52s** (~92% reduction).
- **Async_chunk also improves E2E latency and RTF**  
  Example: concurrency 1, E2E improves by ~6% (**6.58s → 6.18s**); concurrency 10 improves by ~17% (**13.52s → 11.15s**).  
  RTF improves (e.g., ~8% at conc 1: **0.24 → 0.22**, ~16% at conc 10: **0.49 → 0.41**).
- **Code2Wav batching (batch size 64 vs 1) helps when async_chunk is off, especially at higher concurrency**  
  Example: concurrency 10, E2E improves by ~30% (**13.5s → 9.5s**), and TTFP improves similarly (**13.4s → 9.4s**).

### Plots (TTFP and RTF)

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/performance/qwen3-omni_ttfp_performance.png">
    <img alt="TTFP Performance Data Comparison" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/performance/qwen3-omni_ttfp_performance.png" width=100%>
  </picture>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/performance/qwen3-omni_rft_performance.png">
    <img alt="RTF Performance Data Comparison" src="https://raw.githubusercontent.com/vllm-project/vllm-omni/refs/heads/main/docs/source/performance/qwen3-omni_rft_performance.png" width=100%>
  </picture>
</p>

- **Lower first-audio latency:** downstream can begin earlier.
- **Higher stability under concurrency:** waiting requests stop blocking runnable work.
- **Better end-to-end smoothness:** chunk-aware scheduling reduces jitter and avoids bursty handoff.

Async Chunk is not just "send smaller pieces."  
It is a scheduler-and-transfer contract: **forward early, schedule only when ready, and flush tails safely.**

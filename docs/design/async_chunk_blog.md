# Async Chunk: Making Multi-Stage Speech Generation Truly Streamable

In multi-stage multimodal systems like Qwen3-Omni, user-perceived latency is often determined not by a single model, but by the slowest wait in the end-to-end pipeline.  
The core idea of **Async Chunk** is simple: instead of waiting for a full stage output, forward small chunks as soon as they are ready, so the system shifts from serial waiting to pipelined overlap.

---

## Why Async Chunk?

### 1) Serial stage handoff amplifies first-packet latency

Speech generation typically goes through multiple stages (for example, Thinker -> Talker -> Code2Wav).  
If each stage must fully complete before the next stage starts, users experience a long initial silence before hearing anything.

This is a pipeline-level waiting problem, not just a model-speed problem.

### 2) Model-side streaming potential needs system-side support

The models like Qwen3-Omni can start producing useful outputs early, but if serving still uses full-buffer handoff, users cannot feel that benefit in practice.

### 3) At higher concurrency, waiting turns into a system bottleneck

Under concurrent traffic, requests blocked on full upstream outputs can also reduce scheduler efficiency.  
What starts as a per-request first-packet issue quickly becomes a throughput and stability issue for the whole service.

---
## Where It Matters Most

- Real-time voice assistants that care about "time to first sound"
- Multimodal applications that need fast spoken responses
- Online serving environments where first-packet stability is critical at concurrency

---

## Solution Strategy: From Full-Buffer Handoff to Asynchronous Chunk Pipeline

Async Chunk can be understood in three layers:

### Strategy 1: Chunk-level forwarding to start downstream earlier

Downstream stages no longer wait for full upstream completion.  
Once a chunk is available, it is forwarded immediately.  
This allows stage execution to overlap like a pipeline instead of running as a strict relay race.

### Strategy 2: Non-blocking scheduling for chunk-waiting requests

Async Chunk is not only about chunking, but also about asynchronous scheduling.  
Requests waiting for chunks enter a waiting state, while the scheduler continues serving runnable requests, preventing one blocked request from slowing down everyone else.

### Strategy 3: Lightweight chunk aggregation where it helps

For later speech stages (such as Code2Wav), chunk aggregation can be applied before processing to avoid excessive fragmentation overhead.  
This balances two goals: faster perceived response and stable overall throughput.

---

## Results and User Impact

**Async Chunk first improves when users hear the first audio packet, then improves overall system efficiency.**

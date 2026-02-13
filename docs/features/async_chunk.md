# Async Chunk: Making Multi-Stage Speech Generation Truly Streamable

In multi-stage multimodal systems like Qwen3-Omni, user-perceived latency is often determined not by a single model, but by the slowest wait in the end-to-end pipeline.  
The core idea of **Async Chunk** is simple: instead of waiting for a full stage output, forward small chunks as soon as they are ready, so the system shifts from serial waiting to pipelined overlap.

---

## Motivation: Why Async Chunk?

### 1) Serial stage handoff amplifies first-packet latency

Speech generation in Qwen3-Omni typically goes through multiple stages (for example, Thinker -> Talker -> Code2Wav).  
If each stage must fully complete before the next stage starts, users experience a long initial silence before hearing anything.

This is a pipeline-level waiting problem, not just a model-speed problem.

### 2) Model-side streaming potential needs system-side support

The Qwen3-Omni technical report emphasizes low first-packet latency for streaming speech.  
The model can start producing useful outputs early, but if serving still uses full-buffer handoff, users cannot feel that benefit in practice.

In short: **the model can speak early, but the system still delivers late**.

### 3) At higher concurrency, waiting turns into a system bottleneck

Under concurrent traffic, requests blocked on full upstream outputs can also reduce scheduler efficiency.  
What starts as a per-request first-packet issue quickly becomes a throughput and stability issue for the whole service.

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

Based on Qwen3-Omni benchmark data in PR #962 (H800):

- In a representative setup (CUDA Graph disabled), TTFP dropped from about **169s** to about **2.2s** (about **98.7%** reduction), with meaningful end-to-end latency improvement as well (about **21.3%**).
- With CUDA Graph enabled, TTFP dropped from about **30s** to about **0.7s** (about **97.6%** reduction).

The key takeaway is straightforward:  
**Async Chunk first improves when users hear the first audio packet, then improves overall system efficiency.**

> Note: exact gains depend on model configuration, concurrency level, and hardware setup.

---

## Where It Matters Most

- Real-time voice assistants that care about "time to first sound"
- Multimodal applications that need fast spoken responses
- Online serving environments where first-packet stability is critical at concurrency

---

## One-Sentence Summary

**Async Chunk is not about making the model more complex; it is about making multi-stage serving truly stream-oriented: forward early, overlap execution, and prioritize first-packet user experience.**

---

## References

- PR design doc: <https://github.com/vllm-project/vllm-omni/pull/962>
- Paper: <https://arxiv.org/pdf/2509.17765>
- Style references:  
  - <https://blog.vllm.ai/2025/12/19/vllm-omni-diffusion-cache-acceleration.html>  
  - <https://blog.vllm.ai/2025/12/15/vllm-epd.html>

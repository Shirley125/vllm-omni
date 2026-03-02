# Qwen3-Omni 在 vLLM-Omni 上的支持与性能优化实践（草稿）

> 本文为可发布博客草稿，按 vLLM blog 风格组织。  
> 所有实验数据与图片位置已预留，你可以后续直接替换“待补充”内容。

## 1. 总结

Qwen3-Omni 目前已经可以在 vLLM-Omni 上稳定运行，并可通过以下能力实现性能优化和流式体验增强：

- **Batching（跨 stage 批量推理）**
- **CUDA Graph（Thinker / Talker 图捕获）**
- **Text Streaming Output（文本流式输出）**
- **Audio Streaming Output（音频流式输出）**
- **Async Chunk（跨 stage 异步分块流水）**

相较于 Transformers 基线，在同等硬件和请求配置下，vLLM-Omni 的整体性能提升为：

- **吞吐提升：`[待补充：xx%]`**
- **E2E 时延下降：`[待补充：xx%]`**
- **TTFP 改善：`[待补充：xx%]`**

---

## 2. Batching 支持

### 2.1 三阶段（Thinker、Talker & Talker-MTP、Code2Wav）批量推理原理

Qwen3-Omni 在 vLLM-Omni 中采用三阶段架构：

- Stage 0：**Thinker**（多模态理解 + 文本生成）
- Stage 1：**Talker + Talker-MTP**（文本到 codec 表示）
- Stage 2：**Code2Wav**（codec 到波形）

Batching 的核心是：在每个 stage 内将可合并请求打包执行，并在 stage 间通过连接器传递结果，以减少单请求调度和 kernel 启动开销。

1) **Thinker / Talker 批量执行**  
   在 stage worker 中，按 `runtime.max_batch_size` 与时间窗口聚合请求，再统一调用 `generate`。  
   参考：
   - PR #438: [Support Qwen Omni online batch inference](https://github.com/vllm-project/vllm-omni/pull/438)

2) **Talker-MTP / Code Predictor 批量解码**  
   Talker 的解码链路中引入对 code predictor 的 batch 推理能力。  
   参考：
   - Issue #420: [Code_predictor Support batch inference](https://github.com/vllm-project/vllm-omni/issues/420)
   - PR #456: [Support code_predictor batch inference](https://github.com/vllm-project/vllm-omni/pull/456)

3) **Code2Wav 批量生成**  
   Code2Wav 阶段支持 batch 化音频解码，并按请求切分输出，缓解高并发下 stage2 瓶颈。  
   参考：
   - RFC #1211: [Qwen3 Omni code2wav stage support batching](https://github.com/vllm-project/vllm-omni/issues/1211)
   - PR #1246: [Support Qwen3 Omni code2wav batch infernce with async chunk](https://github.com/vllm-project/vllm-omni/pull/1246)

### 2.2 各 Stage 开启 Batching 前后性能对比（待补充）

> 指标：E2E、TTFP、RTF。  
> 测试条件（硬件、并发、输入长度、输出长度）请在你补数据时一起写入。

| Stage | 配置 | E2E (ms) | TTFP (ms) | RTF | 相对提升 |
|---|---|---:|---:|---:|---:|
| Thinker | Batching Off | 待补充 | 待补充 | 待补充 | - |
| Thinker | Batching On | 待补充 | 待补充 | 待补充 | 待补充 |
| Talker & Talker-MTP | Batching Off | 待补充 | 待补充 | 待补充 | - |
| Talker & Talker-MTP | Batching On | 待补充 | 待补充 | 待补充 | 待补充 |
| Code2Wav | Batching Off | 待补充 | 待补充 | 待补充 | - |
| Code2Wav | Batching On | 待补充 | 待补充 | 待补充 | 待补充 |

---

## 3. CUDA Graph 支持

### 3.1 CUDA Graph 原理简介

CUDA Graph 通过捕获稳定形状下的一组 GPU 操作并重复回放，减少 CPU 侧 launch 开销和调度抖动，通常对高频 decode 场景收益明显。

在 Qwen3-Omni 的 vLLM-Omni 实践中，CUDA Graph 已支持：

- **Thinker 阶段**
- **Talker 阶段（含 Talker-MTP 相关路径）**

参考：

- PR #523: [Support Qwen3 Omni thinker cuda graph](https://github.com/vllm-project/vllm-omni/pull/523)
- PR #669: [Support Qwen3 Omni talker cudagraph](https://github.com/vllm-project/vllm-omni/pull/669)

工程侧常见开关方式为在 stage config 中对对应 stage 设置 `enforce_eager: false`（启用图捕获路径），并结合 batching 获取更稳定收益。

### 3.2 叠加 Batching 后，CUDA Graph 开关对比（待补充）

| 配置 | E2E (ms) | TTFP (ms) | RTF | 备注 |
|---|---:|---:|---:|---|
| Batching On + CUDA Graph Off | 待补充 | 待补充 | 待补充 | `enforce_eager: true` |
| Batching On + CUDA Graph On | 待补充 | 待补充 | 待补充 | `enforce_eager: false` |
| 相对变化 | 待补充 | 待补充 | 待补充 | 待补充 |

---

## 4. Text Streaming Output 和 Audio Streaming Output 支持

### 4.1 Streaming Output 原理简介及对 TTFT/TTFP 的影响

Streaming 的核心收益是“边生成边返回”：

- **Text Streaming**：token 级增量返回，显著降低 TTFT（首 token 时间）体感。
- **Audio Streaming**：音频 chunk 增量返回，显著降低 TTFP（首包时间）体感。

参考：

- PR #367: [Basic version of supporting streaming output](https://github.com/vllm-project/vllm-omni/pull/367)
- Audio streaming output：在 Qwen3-Omni 实践里可结合 stage2 调度参数（如 `max_num_batched_tokens`）进行吞吐/首包延迟权衡。

实践建议：

- 若优先首包体验（TTFP），可尝试降低 stage2 的 batch 聚合强度（例如更保守的 `max_num_batched_tokens`）。
- 若优先吞吐，可提高 stage2 batch 聚合强度，但首包延迟可能上升。

### 4.2 叠加 Batching + CUDA Graph 后，Streaming 开关对比（待补充）

> 指标：E2E、TTFT、TTFP、RTF。

| 配置 | E2E (ms) | TTFT (ms) | TTFP (ms) | RTF | 备注 |
|---|---:|---:|---:|---:|---|
| Streaming Off | 待补充 | 待补充 | 待补充 | 待补充 | 仅最终返回 |
| Text Streaming On, Audio Streaming Off | 待补充 | 待补充 | 待补充 | 待补充 | 文本先返回 |
| Text Streaming On, Audio Streaming On | 待补充 | 待补充 | 待补充 | 待补充 | 文本+音频都流式 |
| 相对变化 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |

---

## 5. Async Chunk 支持

### 5.1 Async Chunk 原理简介（含图位占位）

Async Chunk 的核心思想是：将跨 stage 的“整请求串行传递”改为“按 chunk 异步传递”，让 Thinker / Talker / Code2Wav 尽早并行重叠执行。

在 Qwen3-Omni 中，典型路径为：

- Thinker → Talker：按解码步持续传递中间表示
- Talker → Code2Wav：按 codec chunk 传递
- Code2Wav：按 chunk 解码并可直接流式输出

参考：

- PR #727: [Support async computation and communication across stages by chunks](https://github.com/vllm-project/vllm-omni/pull/727)
- RFC #268: [Support async computation and communication across stages by chunks](https://github.com/vllm-project/vllm-omni/issues/268)
- 设计文档：`docs/design/feature/async_chunk_design.md`

**图位占位（待补充图片）**

> **图 1（待补充）**：同步串行流水 vs Async Chunk 流水示意图（重点突出 stage 间重叠执行）。  
> **图 2（待补充）**：TTFP 对比曲线（Async Chunk Off/On，多并发点位）。  
> **图 3（待补充）**：E2E 与 RTF 对比柱状图（叠加 batching + cuda graph + streaming 输入输出）。

### 5.2 叠加 CUDA Graph + Batching + Streaming Input/Output 后，Async Chunk 开关对比（待补充）

| 配置 | E2E (ms) | TTFP (ms) | RTF | 备注 |
|---|---:|---:|---:|---|
| Async Chunk Off | 待补充 | 待补充 | 待补充 | 已开启 batching + cuda graph + streaming input/output |
| Async Chunk On | 待补充 | 待补充 | 待补充 | 已开启 batching + cuda graph + streaming input/output |
| 相对变化 | 待补充 | 待补充 | 待补充 | 待补充 |

---

## 6. Qwen3-Omni 推理实践

下面给出一组可直接复现的实践步骤，覆盖如何部署以及如何开启 batching、cuda graph、text/audio streaming、async chunk。

### 6.1 启动服务（默认三阶段 + batching + thinker/talker cudagraph）

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni \
  --port 8091 \
  --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_omni_moe.yaml
```

说明：

- `runtime.max_batch_size` 控制各 stage 请求批大小。
- Thinker / Talker 设置 `enforce_eager: false` 可走 CUDA Graph 路径。
- Code2Wav 默认 `enforce_eager: true`。

### 6.2 启动服务（开启 Async Chunk）

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni \
  --port 8091 \
  --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_omni_moe_async_chunk.yaml
```

说明：

- 顶层 `async_chunk: true`。
- 使用 `custom_process_next_stage_input_func`（thinker2talker_async_chunk / talker2code2wav_async_chunk）实现分块传递。

### 6.3 客户端开启 Text / Audio Streaming

```bash
cd examples/online_serving/qwen3_omni

# 流式返回（文本 + 音频）
python openai_chat_completion_client_for_multimodal_generation.py \
  --query-type use_image \
  --stream
```

可选：

- `--modalities text`：仅文本输出
- 默认或 `--modalities audio`：文本 + 音频输出

### 6.4 关键配置项速查（建议在自定义 stage config 中显式声明）

```yaml
async_chunk: true  # 是否启用跨 stage 异步分块
stage_args:
  - stage_id: 0 # thinker
    runtime:
      max_batch_size: 64   # batching
    engine_args:
      enforce_eager: false # cudagraph on
      max_num_batched_tokens: 32768
      custom_process_next_stage_input_func: vllm_omni.model_executor.stage_input_processors.qwen3_omni.thinker2talker_async_chunk

  - stage_id: 1 # talker
    runtime:
      max_batch_size: 64
    engine_args:
      enforce_eager: false # cudagraph on
      max_num_batched_tokens: 32768
      custom_process_next_stage_input_func: vllm_omni.model_executor.stage_input_processors.qwen3_omni.talker2code2wav_async_chunk

  - stage_id: 2 # code2wav
    runtime:
      max_batch_size: 64
    engine_args:
      enforce_eager: true
      max_num_batched_tokens: 51200 # 可用于吞吐/TTFP 权衡调优
```

### 6.5 建议的实验补数顺序（便于后续补图补表）

1. **Transformers 基线**：固定输入集合与并发，记录 E2E/TTFT/TTFP/RTF。  
2. **vLLM-Omni 基线**：仅开 batching。  
3. **+ CUDA Graph**：仅切换 `enforce_eager`。  
4. **+ Streaming**：先 text streaming，再 text+audio streaming。  
5. **+ Async Chunk**：最终叠加方案，补齐全指标与对比图。

---

## 参考链接

- 目标风格参考（vLLM blog）：  
  <https://blog.vllm.ai/2026/02/13/gb300-deepseek.html>
- vLLM-Omni 仓库：  
  <https://github.com/vllm-project/vllm-omni>
- 本文相关 PR / RFC：
  - #438: <https://github.com/vllm-project/vllm-omni/pull/438>
  - #420: <https://github.com/vllm-project/vllm-omni/issues/420>
  - #456: <https://github.com/vllm-project/vllm-omni/pull/456>
  - #1211: <https://github.com/vllm-project/vllm-omni/issues/1211>
  - #1246: <https://github.com/vllm-project/vllm-omni/pull/1246>
  - #523: <https://github.com/vllm-project/vllm-omni/pull/523>
  - #669: <https://github.com/vllm-project/vllm-omni/pull/669>
  - #367: <https://github.com/vllm-project/vllm-omni/pull/367>
  - #727: <https://github.com/vllm-project/vllm-omni/pull/727>
  - #268: <https://github.com/vllm-project/vllm-omni/issues/268>

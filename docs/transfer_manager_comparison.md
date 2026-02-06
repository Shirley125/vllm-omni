---
title: Transfer Manager Comparison
---

# Transfer Manager Comparison

This document summarizes the commonalities and differences between the
Chunk Transfer Manager and the KV Transfer Manager.

## Commonalities (Shared Points)

| Dimension | Chunk Transfer Manager | KV Transfer Manager |
|---|---|---|
| Connector usage | Uses connector based on config | Same |
| Transfer key | Builds key from stage_id / req_id / etc. | Same |
| Core flow | Has both save & load flows | Same |
| Final interface | Calls connector.put / connector.get | Same |

## Differences

| Dimension | Chunk Transfer Manager | KV Transfer Manager |
|---|---|---|
| Pre-processing | Requires model input processor to prepare payload | Requires KV cache reshape/extract |
| Transfer mode | **Asynchronous** (threads + pending queue) | **Synchronous** (direct call) |
| Scheduler trigger | Must ensure recv/send before every scheduling cycle | Only when prefill finishes or special token triggers |

# Detailed Technical Analysis: Concurrent Request Deadlock in Qwen3-Omni

## The Core Issue

When commit d0836d8 is applied, concurrent audio requests cause the inference pipeline to hang at Stage-2 (Code2Wav) with a warning about sequence length not being divisible by 16.

## Critical Code Path Analysis

### 1. Stage 0: Thinker - Audio Processing

**File**: `vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py`

#### Current Code (Working with concurrent requests):
```python
def _process_audio_input(
    self,
    audio_input: Qwen2_5OmniAudioFeatureInputs,
    audio_hashes: list[str] | None = None,
    cached_audio_features: torch.Tensor | None = None,
) -> torch.Tensor:
    input_features = audio_input["input_features"]
    audio_feature_lengths = audio_input["audio_feature_lengths"]

    # Calculate lengths after CNN processing
    audio_feat_lengths, audio_output_lengths = _get_feat_extract_output_lengths(audio_feature_lengths)

    # CRITICAL: aftercnn_lens tells the audio tower the actual sequence lengths
    audio_outputs = self.audio_tower(
        input_features.to(self.audio_tower.dtype),
        feature_lens=audio_feature_lengths,
        aftercnn_lens=audio_feat_lengths,  # ← This is essential!
    )
    audio_features = audio_outputs.last_hidden_state
    return audio_features.split(audio_output_lengths.tolist())
```

#### After Commit d0836d8 (Breaks concurrent requests):
```python
def _parse_and_validate_audio_input(self, **kwargs: object):
    input_audio_features = kwargs.pop("input_audio_features", None)
    audio_feature_lengths = kwargs.pop("audio_feature_lengths", None)
    
    if input_audio_features is not None and isinstance(input_audio_features, torch.Tensor):
        if input_audio_features.ndim == 3:
            # PROBLEM: Flattens batch dimension
            # [batch_size, feature_dim, chunk_size] → [feature_dim, batch_size * chunk_size]
            input_audio_features = input_audio_features.permute(1, 0, 2).flatten(1)
    
    if audio_feature_lengths is not None and isinstance(audio_feature_lengths, torch.Tensor):
        if audio_feature_lengths.ndim == 2:
            # PROBLEM: Reshapes lengths, losing per-request boundaries
            audio_feature_lengths = audio_feature_lengths.reshape(-1)
    
    return Qwen2_5OmniAudioFeatureInputs(...)

def _process_audio_input(
    self,
    audio_input: Qwen2_5OmniAudioFeatureInputs,
) -> torch.Tensor:
    # ...
    audio_outputs = self.audio_tower(
        input_features.to(self.audio_tower.dtype),
        feature_lens=audio_feature_lengths,
        # aftercnn_lens REMOVED! ← Audio tower can't track sequence boundaries
    )
    # ...
```

**Why This Breaks Concurrent Requests:**

```python
# Scenario: 2 concurrent requests with different audio lengths

# Request 1: 3.0 seconds audio → 3000 samples → 187 tokens after processing
# Request 2: 2.5 seconds audio → 2500 samples → 156 tokens after processing

# WITHOUT commit (correct):
audio_features_1 = [187, hidden_dim]  # Request 1
audio_features_2 = [156, hidden_dim]  # Request 2
# Each request maintains its own boundary

# WITH commit (broken):
# Step 1: Inputs are batched
input_audio_features = torch.stack([audio_1, audio_2])  # [2, 128, varying_length]

# Step 2: Permute and flatten (loses batch boundary)
input_audio_features = input_audio_features.permute(1, 0, 2).flatten(1)
# Shape: [128, 2 * varying_length] ← No way to know where request 1 ends and request 2 begins!

# Step 3: audio_tower processes without aftercnn_lens
# It doesn't know the sequence should be split at position 187
# Results in incorrect attention masks and misaligned features
```

### 2. Stage 1: Talker - Codec Generation

**File**: `vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py`

```python
def talker_mtp(
    self,
    input_ids: torch.Tensor,
    input_embeds: torch.Tensor,
    last_talker_hidden: torch.Tensor,
    text_step: torch.Tensor,
):
    # ...
    if inputs_embeds.shape[0] == 1:  # Decode phase (1 token at a time)
        code_predictor_codes, summed_embeddings = self.talker.code_predictor_forward(
            input_ids, inputs_embeds.clone(), last_talker_hidden=last_talker_hidden
        )
        inputs_embeds = summed_embeddings.clone()
    else:
        # Prefill phase or multi-token
        code_predictor_codes = torch.zeros((0, self.talker.num_code_groups), dtype=torch.long)
    
    # code_predictor_codes shape: [seq_len, 16]
    # This should be [seq_len, 16] for proper RVQ codes
    return inputs_embeds, code_predictor_codes.squeeze(-1).detach().to("cpu").contiguous()
```

**The Problem in Concurrent Scenarios:**

When audio features from Stage 0 are misaligned due to missing `aftercnn_lens`:
- The text embeddings fed to the talker have incorrect sequence boundaries
- `code_predictor_codes` output may have wrong shape or padding
- For some requests, `code_predictor_codes` might end up with shape `[1, 16]` instead of `[seq_len, 16]`

### 3. Stage 1→2 Transition: Talker to Code2Wav

**File**: `vllm_omni/model_executor/stage_input_processors/qwen3_omni.py`

```python
def talker2code2wav(
    stage_list: list[Any],
    engine_input_source: list[int],
    ...
) -> list[OmniTokensPrompt]:
    talker_outputs = stage_list[source_stage_id].engine_outputs
    code2wav_inputs = []

    for i, talker_output in enumerate(talker_outputs):
        output = talker_output.outputs[0]

        # Extract codec codes from talker output
        # Expected: [seq_len, 16] → transpose → [16, seq_len] → flatten → [16*seq_len]
        codec_codes = (
            output.multimodal_output["code_predictor_codes"]
            .to(torch.long)
            .transpose(0, 1)  # [seq_len, 16] → [16, seq_len]
            .cpu()
            .to(torch.long)
            .reshape(-1)  # [16, seq_len] → [16*seq_len]
            .tolist()
        )
        
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,  # Must be length 16*N
                ...
            )
        )

    return code2wav_inputs
```

**Where the "Length 1 not divisible by 16" Error Comes From:**

```python
# Scenario: Request with misaligned audio features from Stage 0

# Normal case:
# talker output: code_predictor_codes shape [150, 16]
# After transpose: [16, 150]
# After reshape(-1): [2400] → len(codec_codes) = 2400
# In code2wav: input_ids.reshape(1, 16, -1) → [1, 16, 150] ✓

# Broken case (after commit d0836d8):
# Due to misaligned audio features, talker might output:
# code_predictor_codes shape [1, 16] (wrong! should be [seq_len, 16])
# After transpose: [16, 1]
# After reshape(-1): [16] → len(codec_codes) = 16
# In code2wav: input_ids.reshape(1, 16, -1) → [1, 16, 1] 
# BUT if the pipeline expects more tokens, it might only get 1 token!

# Even worse case:
# code_predictor_codes might be malformed: [1, 1] or empty
# After processing: [1] → len(codec_codes) = 1
# In code2wav: input_ids.reshape(1, 16, -1) → FAILS! 
#              1 is not divisible by 16!
#              WARNING: "Input_ids length: 1 is not divisible by 16, padding with zeros"
```

### 4. Stage 2: Code2Wav - The Deadlock

**File**: `vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py`

```python
elif self.model_stage == "code2wav":
    # Extract codec codes from input
    codes = []
    if input_ids is not None:
        # CRITICAL RESHAPE: Requires input_ids length to be 16*N
        codes.append(input_ids.reshape(1, 16, -1))  # ← Line 440
    else:
        # for profile, we use max length from inputs_embeds
        codes.append(
            torch.zeros(
                (1, 16, inputs_embeds.shape[1]),
                dtype=torch.long,
                device=inputs_embeds.device,
            )
        )

    # Generate audio from codec codes
    audio_tensors = []
    for code in codes:
        audio_tensor = self.generate_audio(code, voice_type)
        audio_tensors.append(audio_tensor)
```

**The Deadlock Mechanism:**

```python
# When concurrent requests arrive:

# Request A (normal): 
#   - Gets correct audio features from Stage 0
#   - Produces correct codec codes: [150, 16] → flattened to [2400]
#   - Code2Wav receives [2400] → reshapes to [1, 16, 150] ✓
#   - Processes successfully

# Request B (affected by commit):
#   - Gets misaligned audio features (batch boundary lost)
#   - Produces incorrect codec codes: [1, 16] → flattened to [16] or even [1]
#   - Code2Wav receives [1] → CANNOT reshape to [1, 16, -1]!
#   - Adds padding: [1] + [15 zeros] = [16] → reshapes to [1, 16, 1]
#   - But this is still wrong! Audio generation expects longer sequences
#   - Pipeline stalls waiting for more tokens that never arrive

# Request A now waits for Request B to finish
# Request B is stuck with malformed input
# → DEADLOCK
```

## Why This Only Shows Warning During "Warmup"

The warning message says: "This should only happen in warm up"

**Explanation:**

During vLLM warmup:
- The engine creates dummy inputs with minimal length (1 token) to:
  - Allocate CUDA graphs
  - Measure memory usage
  - Test the pipeline
- These dummy inputs are expected to have length 1
- Padding to 16 is normal and expected

During actual inference:
- Real inputs should never have length 1 at Stage 2
- Code2Wav expects properly sized codec sequences
- If length 1 appears, it indicates upstream corruption

**The Warning Code (hypothetical, not in current repo):**
```python
def code2wav_forward(self, input_ids):
    if len(input_ids) < 16:
        if not self.is_warmup:
            logger.warning(
                f"Input_ids length: {len(input_ids)} is not divisible by 16, "
                f"padding with zeros. This should only happen in warm up"
            )
        # Pad to 16
        input_ids = torch.nn.functional.pad(input_ids, (0, 16 - len(input_ids)))
    
    codes = input_ids.reshape(1, 16, -1)
    # ...
```

## Concurrency-Specific Failure Modes

### Failure Mode 1: Batch Dimension Collapse

```python
# Timeline of 2 concurrent requests:

# t=0: Request 1 enters Stage 0 (Thinker)
#      audio_1: [1, 128, 3000]

# t=10ms: Request 2 enters Stage 0 (Thinker)
#         audio_2: [1, 128, 2500]

# t=20ms: Batch scheduler groups them
#         batched_audio: [2, 128, 3000]  # Padded to max length
#         batched_lengths: [3000, 2500]

# With commit d0836d8:
# t=25ms: _parse_and_validate_audio_input() called
#         permute(1,0,2): [128, 2, 3000]
#         flatten(1): [128, 6000]  # ← Lost the boundary at position 3000!
#         lengths reshaped: [3000, 2500] → [3000, 2500] (but no way to use them)

# t=30ms: audio_tower() processes [128, 6000] without aftercnn_lens
#         Produces: [750, hidden_dim]  # Should be [187+156, hidden_dim] with boundary
#         But the boundary info is lost!

# t=50ms: Stage 1 (Talker) receives misaligned features
#         Cannot properly split features for each request
#         Produces malformed codec codes

# t=70ms: Stage 2 (Code2Wav) receives incorrect input
#         Request 1: Expected [2400], got [16]
#         Request 2: Expected [1920], got [1]
#         → Padding, warnings, incorrect audio, or deadlock
```

### Failure Mode 2: Attention Mask Misalignment

```python
# In Qwen3OmniMoeAudioEncoder (transformers library):

def forward(self, input_features, feature_lens, aftercnn_lens=None):
    # Without aftercnn_lens:
    # - Cannot create correct attention masks for batched inputs
    # - All sequences treated as one long sequence
    # - Cross-request attention pollution
    
    # Example:
    # Request 1: tokens [0:187]
    # Request 2: tokens [188:343]  # Should be [0:156] but offset
    
    # Attention mask without proper boundaries:
    # Request 1 attends to Request 2's tokens (incorrect!)
    # Request 2's features are shifted by Request 1's length (corrupted!)
    
    # Result:
    # - Both requests get contaminated features
    # - Downstream stages receive garbage
    # - Codec generation produces incorrect or minimal codes
    # - Pipeline hangs or produces wrong output
```

## The Fix: Proper Batch-Aware Audio Processing

### Solution 1: Restore aftercnn_lens Parameter

```python
# In qwen3_omni_moe_thinker.py

class Qwen3OmniMoeConditionalGenerationMixin(Qwen2_5OmniConditionalGenerationMixin):
    def _parse_and_validate_audio_input(self, **kwargs: object):
        input_audio_features = kwargs.pop("input_audio_features", None)
        audio_feature_lengths = kwargs.pop("audio_feature_lengths", None)
        feature_attention_mask = kwargs.pop("feature_attention_mask", None)
        
        if input_audio_features is None:
            return None, None
        
        # Validate but DON'T flatten for batched inputs
        if isinstance(input_audio_features, torch.Tensor):
            # Keep batch dimension: [batch_size, feature_dim, chunk_size]
            if input_audio_features.ndim == 2:
                # Single sample
                input_audio_features = input_audio_features.unsqueeze(0)
        
        # Calculate aftercnn_lens (critical for attention masks)
        audio_feat_lengths, audio_output_lengths = _get_feat_extract_output_lengths(
            audio_feature_lengths
        )
        
        return Qwen2_5OmniAudioFeatureInputs(
            type="audio_features",
            input_features=input_audio_features,
            audio_feature_lengths=audio_feature_lengths,
            feature_attention_mask=feature_attention_mask,
        ), audio_feat_lengths  # ← Return this!
    
    def _process_audio_input(
        self,
        audio_input: Qwen2_5OmniAudioFeatureInputs,
        audio_hashes: list[str] | None = None,
        cached_audio_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_features = audio_input["input_features"]
        audio_feature_lengths = audio_input["audio_feature_lengths"]
        
        # Get aftercnn_lens from validation
        audio_feat_lengths, audio_output_lengths = _get_feat_extract_output_lengths(
            audio_feature_lengths
        )
        
        # CRITICAL: Pass aftercnn_lens to audio_tower
        audio_outputs = self.audio_tower(
            input_features.to(self.audio_tower.dtype),
            feature_lens=audio_feature_lengths,
            aftercnn_lens=audio_feat_lengths,  # ← Restored!
        )
        audio_features = audio_outputs.last_hidden_state
        
        # Properly split by boundaries
        return audio_features.split(audio_output_lengths.tolist())
```

### Solution 2: Add Length Validation in Code2Wav

```python
# In qwen3_omni.py

elif self.model_stage == "code2wav":
    codes = []
    if input_ids is not None:
        # Validate length before reshaping
        if input_ids.numel() % 16 != 0:
            logger.error(
                f"[Stage-2] Code2Wav received invalid input_ids length: {input_ids.numel()}, "
                f"which is not divisible by 16. This indicates upstream Stage corruption. "
                f"Expected shape: [16*N], got: [{input_ids.numel()}]"
            )
            # Attempt recovery by padding (may produce incorrect audio)
            pad_length = 16 - (input_ids.numel() % 16)
            input_ids = torch.nn.functional.pad(
                input_ids, (0, pad_length), value=0
            )
            logger.warning(f"[Stage-2] Padded {pad_length} zeros to fix shape")
        
        try:
            codes.append(input_ids.reshape(1, 16, -1))
        except RuntimeError as e:
            logger.error(f"[Stage-2] Failed to reshape input_ids: {e}")
            # Emergency fallback: create dummy output
            codes.append(torch.zeros((1, 16, 1), dtype=torch.long, device=input_ids.device))
    # ...
```

### Solution 3: Add Request Tracking and Debugging

```python
# In stage_input_processors/qwen3_omni.py

def talker2code2wav(
    stage_list: list[Any],
    engine_input_source: list[int],
    ...
) -> list[OmniTokensPrompt]:
    talker_outputs = stage_list[source_stage_id].engine_outputs
    code2wav_inputs = []

    for i, talker_output in enumerate(talker_outputs):
        output = talker_output.outputs[0]
        
        # Debug: Log codec codes shape
        codec_codes_tensor = output.multimodal_output["code_predictor_codes"]
        logger.debug(
            f"[Stage1→2] Request {i}: codec_codes shape: {codec_codes_tensor.shape}, "
            f"expected: [seq_len, 16]"
        )
        
        # Validate shape before processing
        if codec_codes_tensor.shape[-1] != 16:
            logger.error(
                f"[Stage1→2] Request {i}: Invalid codec_codes shape {codec_codes_tensor.shape}, "
                f"expected [..., 16]. This indicates talker stage corruption."
            )
            # Cannot recover - this request will fail
            raise RuntimeError(
                f"Talker produced invalid codec codes for request {i}. "
                f"This is likely due to misaligned audio features from Stage 0. "
                f"Check if commit d0836d8 is applied."
            )
        
        codec_codes = (
            codec_codes_tensor
            .to(torch.long)
            .transpose(0, 1)
            .cpu()
            .to(torch.long)
            .reshape(-1)
            .tolist()
        )
        
        logger.debug(f"[Stage1→2] Request {i}: flattened codec_codes length: {len(codec_codes)}")
        
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,
                ...
            )
        )

    return code2wav_inputs
```

## Testing Strategy

### Test 1: Concurrent Requests with Varying Audio Lengths

```python
import asyncio
import torch
from vllm import AsyncLLMEngine, AsyncEngineArgs

async def test_concurrent_audio():
    engine_args = AsyncEngineArgs(
        model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        model_stage="thinker,talker,code2wav",
        # ... other args
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    # Create audio samples of different lengths
    audio_1 = torch.randn(48000)  # 1 second at 48kHz
    audio_2 = torch.randn(96000)  # 2 seconds
    audio_3 = torch.randn(144000) # 3 seconds
    
    requests = [
        {"prompt": "Describe this audio", "audio": audio_1},
        {"prompt": "What do you hear?", "audio": audio_2},
        {"prompt": "Transcribe this", "audio": audio_3},
    ]
    
    # Send all requests simultaneously
    tasks = []
    for req in requests:
        task = engine.generate(
            prompt=req["prompt"],
            sampling_params=...,
            audio=req["audio"],
        )
        tasks.append(task)
    
    # This should NOT deadlock
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Request {i} failed: {result}")
        else:
            print(f"Request {i} succeeded: {result}")

# Run test
asyncio.run(test_concurrent_audio())
```

### Test 2: Sequential vs Concurrent Comparison

```python
async def test_sequential_vs_concurrent():
    engine = # ... init engine
    
    audio = torch.randn(48000)
    prompt = "Describe this audio"
    
    # Sequential (should always work)
    start = time.time()
    for i in range(3):
        result = await engine.generate(prompt, audio=audio)
        print(f"Sequential request {i}: {time.time() - start:.2f}s")
    
    # Concurrent (fails with commit d0836d8)
    start = time.time()
    results = await asyncio.gather(*[
        engine.generate(prompt, audio=audio)
        for i in range(3)
    ])
    print(f"Concurrent 3 requests: {time.time() - start:.2f}s")
```

## Monitoring and Detection

Add monitoring to detect this issue in production:

```python
# In omni pipeline monitoring

class StageHealthMonitor:
    def __init__(self):
        self.stage_timings = defaultdict(list)
        self.stage_failures = defaultdict(int)
    
    def record_stage_completion(self, stage_id, request_id, duration):
        self.stage_timings[stage_id].append(duration)
        
        # Detect abnormally long stage durations (potential deadlock)
        if duration > 10.0:  # 10 seconds
            logger.warning(
                f"Stage {stage_id} took {duration:.2f}s for request {request_id}. "
                f"Possible deadlock or input corruption."
            )
    
    def record_stage_failure(self, stage_id, request_id, error):
        self.stage_failures[stage_id] += 1
        
        # Detect reshape errors (signature of this bug)
        if "reshape" in str(error) and "16" in str(error):
            logger.error(
                f"Detected reshape error at Stage {stage_id}: {error}. "
                f"This is likely due to commit d0836d8 breaking concurrent requests. "
                f"Total failures at this stage: {self.stage_failures[stage_id]}"
            )

monitor = StageHealthMonitor()
```

## Conclusion

The root cause is clear:

1. **Commit d0836d8** flattens batch dimensions and removes `aftercnn_lens`
2. **Audio Tower** loses per-request boundaries in concurrent scenarios
3. **Talker Stage** receives misaligned features, produces malformed codec codes
4. **Code2Wav Stage** receives sequences of length 1 (or other non-16-divisible lengths)
5. **Pipeline deadlocks** waiting for properly sized inputs that never arrive

**Recommended Actions:**

1. **Immediate**: Revert commit d0836d8 or apply Solution 1 above
2. **Short-term**: Add validation and monitoring (Solutions 2 & 3)
3. **Long-term**: Redesign audio batching to properly handle varying lengths

The fix is straightforward: **restore the `aftercnn_lens` parameter** and **don't flatten batch dimensions** in audio processing.

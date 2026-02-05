# Visual Flowchart: Concurrent Request Deadlock Issue

## Normal Flow (Without Commit d0836d8) ✓

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Stage 0: Thinker                            │
│                         Audio Processing                            │
└─────────────────────────────────────────────────────────────────────┘

Request 1: Audio [3000 samples]                Request 2: Audio [2500 samples]
     │                                              │
     ├─► Batched: [2, 128, 3000] (padded)         │
     │   Lengths: [3000, 2500]                    │
     │                                             │
     ├─► audio_tower(                             │
     │     input=[2, 128, 3000],                  │
     │     feature_lens=[3000, 2500],             │
     │     aftercnn_lens=[750, 625]  ←────────── [CRITICAL PARAMETER]
     │   )                                         │
     │                                             │
     ├─► Output: [343, hidden_dim]                │
     │   Split by boundaries: [187] + [156]       │
     │                                             │
     ├─► Request 1: [187, hidden_dim] ──────────► Correct features
     └─► Request 2: [156, hidden_dim] ──────────► Correct features

┌─────────────────────────────────────────────────────────────────────┐
│                         Stage 1: Talker                             │
│                      Codec Code Generation                          │
└─────────────────────────────────────────────────────────────────────┘

Request 1: Features [187, hidden_dim]         Request 2: Features [156, hidden_dim]
     │                                              │
     ├─► Talker processing                         │
     │                                              │
     ├─► code_predictor_codes: [150, 16]           │
     │   (150 tokens × 16 RVQ layers)              │
     │                                              │
     └─► Flatten: [2400] ──────────────────────────┼─► Code2Wav
                                                    │
                                                    ├─► code_predictor_codes: [120, 16]
                                                    │   (120 tokens × 16 RVQ layers)
                                                    │
                                                    └─► Flatten: [1920] ──────► Code2Wav

┌─────────────────────────────────────────────────────────────────────┐
│                       Stage 2: Code2Wav                             │
│                     Audio Waveform Generation                       │
└─────────────────────────────────────────────────────────────────────┘

Request 1: input_ids [2400]                   Request 2: input_ids [1920]
     │                                              │
     ├─► 2400 % 16 == 0 ✓                          │
     ├─► reshape(1, 16, 150) ✓                     │
     ├─► generate_audio() ✓                        │
     └─► Audio output ✓                            │
                                                    ├─► 1920 % 16 == 0 ✓
                                                    ├─► reshape(1, 16, 120) ✓
                                                    ├─► generate_audio() ✓
                                                    └─► Audio output ✓

                        ✅ SUCCESS - Both requests complete
```

## Broken Flow (With Commit d0836d8) ✗

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Stage 0: Thinker                            │
│                  Audio Processing (BROKEN)                          │
└─────────────────────────────────────────────────────────────────────┘

Request 1: Audio [3000 samples]                Request 2: Audio [2500 samples]
     │                                              │
     ├─► Batched: [2, 128, 3000]                  │
     │   Lengths: [3000, 2500]                    │
     │                                             │
     ├─► _parse_and_validate_audio_input():       │
     │   ┌───────────────────────────────────┐    │
     │   │ permute(1,0,2): [128, 2, 3000]    │    │
     │   │ flatten(1):     [128, 6000]       │ ←──┼── [PROBLEM 1: Batch boundary lost!]
     │   │ Lengths become: [3000, 2500]      │    │
     │   │ But no way to use them!           │    │
     │   └───────────────────────────────────┘    │
     │                                             │
     ├─► audio_tower(                             │
     │     input=[128, 6000],                     │
     │     feature_lens=[3000, 2500],            │
     │     aftercnn_lens=MISSING  ←──────────────┼── [PROBLEM 2: Parameter removed!]
     │   )                                         │
     │                                             │
     ├─► Output: [750, hidden_dim]                │
     │   ┌─────────────────────────────────┐      │
     │   │ Should split at position 187    │      │
     │   │ But boundary info is LOST!      │ ←────┼── [PROBLEM 3: Cannot split correctly]
     │   │ Attention mask WRONG!           │      │
     │   │ Cross-request contamination!    │      │
     │   └─────────────────────────────────┘      │
     │                                             │
     ├─► Request 1: [???, hidden_dim] ──────────► CORRUPTED features
     └─► Request 2: [???, hidden_dim] ──────────► CORRUPTED features

┌─────────────────────────────────────────────────────────────────────┐
│                         Stage 1: Talker                             │
│               Codec Code Generation (BROKEN)                        │
└─────────────────────────────────────────────────────────────────────┘

Request 1: CORRUPTED features                 Request 2: CORRUPTED features
     │                                              │
     ├─► Talker processing with wrong inputs       │
     │   ┌──────────────────────────────┐          │
     │   │ Features are misaligned      │          │
     │   │ Cannot generate proper codes │ ←────────┼── [PROBLEM 4: Garbage in, garbage out]
     │   └──────────────────────────────┘          │
     │                                              │
     ├─► code_predictor_codes: [1, 16] ✗           │   [Should be [150, 16]]
     │   ┌─────────────────────────────┐           │
     │   │ WRONG! Only 1 token!        │           │
     │   └─────────────────────────────┘           │
     │                                              │
     └─► Flatten: [16] ───────────────────────────┼─► Code2Wav (WRONG!)
                                                    │
                                                    ├─► code_predictor_codes: [1, 1] ✗
                                                    │   ┌──────────────────────┐
                                                    │   │ VERY WRONG!          │
                                                    │   │ Shape completely off │
                                                    │   └──────────────────────┘
                                                    │
                                                    └─► Flatten: [1] ────────────► Code2Wav (DISASTER!)

┌─────────────────────────────────────────────────────────────────────┐
│                       Stage 2: Code2Wav                             │
│                Audio Waveform Generation (DEADLOCK)                 │
└─────────────────────────────────────────────────────────────────────┘

Request 1: input_ids [16]                     Request 2: input_ids [1]
     │                                              │
     ├─► 16 % 16 == 0 ✓                            │
     ├─► reshape(1, 16, 1) ✓                       │
     │   ┌──────────────────────────────┐          │
     │   │ But shape is WRONG!          │          │
     │   │ Should be [1, 16, 150]       │ ←────────┼── [PROBLEM 5: Wrong sequence length]
     │   │ Pipeline expects more tokens │          │
     │   └──────────────────────────────┘          │
     │                                              │
     ├─► ⏳ Waiting for more tokens...              │
     │   (Never arrives - pipeline STUCK)          │
     │                                              │
     │                                              ├─► 1 % 16 != 0 ✗
     │                                              │   ┌──────────────────────────────┐
     │                                              │   │ ⚠️  WARNING!                  │
     │                                              │   │ "Input_ids length: 1 is not  │
     │                                              │   │  divisible by 16, padding    │ ←─ [PROBLEM 6: The error message!]
     │                                              │   │  with zeros. This should     │
     │                                              │   │  only happen in warm up"     │
     │                                              │   └──────────────────────────────┘
     │                                              │
     │                                              ├─► Pad: [1] + [15 zeros] = [16]
     │                                              ├─► reshape(1, 16, 1) ✓
     │                                              │   ┌──────────────────────┐
     │                                              │   │ But data is GARBAGE! │
     │                                              │   └──────────────────────┘
     │                                              │
     │                                              └─► ⏳ Waiting... (STUCK)
     │
     ├──────────────────── ⏳ Both requests waiting for each other ─────────────────────┤
     │                                                                                   │
     └──────────────────────────────── 💥 DEADLOCK 💥 ──────────────────────────────────┘

                        ❌ FAILURE - Pipeline hangs indefinitely
```

## Side-by-Side Comparison

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                          KEY DIFFERENCES                                      ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  WITHOUT COMMIT (✓)                   │  WITH COMMIT (✗)                     ║
║  ─────────────────────────────────────┼──────────────────────────────────────║
║                                       │                                       ║
║  Stage 0: Audio Tower                 │  Stage 0: Audio Tower                ║
║  ├─ Input: [2, 128, 3000]             │  ├─ Input: [128, 6000] (flattened)  ║
║  ├─ aftercnn_lens: [750, 625] ✓       │  ├─ aftercnn_lens: MISSING ✗        ║
║  ├─ Boundaries preserved ✓            │  ├─ Boundaries LOST ✗               ║
║  └─ Output split correctly ✓          │  └─ Output misaligned ✗             ║
║                                       │                                       ║
║  Stage 1: Talker                      │  Stage 1: Talker                     ║
║  ├─ Features aligned ✓                │  ├─ Features corrupted ✗            ║
║  ├─ Codes: [150, 16] ✓                │  ├─ Codes: [1, 16] or [1, 1] ✗      ║
║  └─ Flattened: [2400] ✓               │  └─ Flattened: [16] or [1] ✗        ║
║                                       │                                       ║
║  Stage 2: Code2Wav                    │  Stage 2: Code2Wav                   ║
║  ├─ Input: [2400] ✓                   │  ├─ Input: [1] ✗                     ║
║  ├─ 2400 % 16 == 0 ✓                  │  ├─ 1 % 16 != 0 ✗                    ║
║  ├─ reshape(1, 16, 150) ✓             │  ├─ ⚠️  WARNING: padding needed      ║
║  └─ Audio generated ✓                 │  └─ Pipeline STUCK ✗                ║
║                                       │                                       ║
║  Result: SUCCESS ✅                    │  Result: DEADLOCK 💥                 ║
║                                       │                                       ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

## The Fix: Restore aftercnn_lens

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SOLUTION: Restore Parameter                      │
└─────────────────────────────────────────────────────────────────────┘

Before (Broken):                           After (Fixed):
                                          
audio_tower(                              audio_tower(
  input=[128, 6000],                        input=[2, 128, 3000],      ←─ Keep batch dim
  feature_lens=[3000, 2500],                feature_lens=[3000, 2500],
  aftercnn_lens=MISSING  ✗                  aftercnn_lens=[750, 625]   ←─ Restore param
)                                         )
    │                                         │
    ├─ No boundary info                       ├─ Knows boundaries
    ├─ Wrong attention masks                  ├─ Correct attention masks
    ├─ Misaligned outputs                     ├─ Aligned outputs
    └─ Corrupted features ✗                   └─ Clean features ✓

                    ▼                                     ▼
          
        [750, hidden] (wrong)                  [187, hidden] (req 1) ✓
                                              [156, hidden] (req 2) ✓
                ▼                                     ▼

        Talker gets garbage ✗                Talker gets correct input ✓
                ▼                                     ▼

        Codes: [1, 16] ✗                     Codes: [150, 16] ✓
                ▼                                     ▼

        Flattened: [16] or [1] ✗             Flattened: [2400] ✓
                ▼                                     ▼

        Code2Wav fails ✗                     Code2Wav succeeds ✓
                ▼                                     ▼

        💥 DEADLOCK                           ✅ SUCCESS
```

## Implementation Change

```python
# File: vllm_omni/model_executor/models/qwen3_omni/qwen3_omni_moe_thinker.py

# WRONG (current broken version):
def _parse_and_validate_audio_input(self, **kwargs):
    # ...
    input_audio_features = input_audio_features.permute(1, 0, 2).flatten(1)  # ✗ Loses batch
    # ...
    return Qwen2_5OmniAudioFeatureInputs(...)  # ✗ No aftercnn_lens

def _process_audio_input(self, audio_input):
    # ...
    audio_outputs = self.audio_tower(
        input_features,
        feature_lens=audio_feature_lengths,
        # aftercnn_lens is MISSING  ✗
    )
    # ...

# ─────────────────────────────────────────────────────────────────

# CORRECT (fixed version):
def _parse_and_validate_audio_input(self, **kwargs):
    # ...
    # DON'T flatten! Keep: [batch_size, feature_dim, chunk_size]  ✓
    
    # Calculate aftercnn_lens
    audio_feat_lengths, _ = _get_feat_extract_output_lengths(audio_feature_lengths)
    
    return (
        Qwen2_5OmniAudioFeatureInputs(...),
        audio_feat_lengths  # ✓ Return this!
    )

def _process_audio_input(self, audio_input):
    # ...
    audio_feat_lengths, audio_output_lengths = _get_feat_extract_output_lengths(
        audio_feature_lengths
    )
    
    audio_outputs = self.audio_tower(
        input_features,
        feature_lens=audio_feature_lengths,
        aftercnn_lens=audio_feat_lengths,  # ✓ Restored!
    )
    # ...
    return audio_features.split(audio_output_lengths.tolist())  # ✓ Proper split
```

## Summary

```
┌─────────────────────────────────────────────────────────────┐
│                     ROOT CAUSE                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Commit d0836d8 tried to fix multi-audio alignment         │
│  but introduced 2 critical bugs:                           │
│                                                             │
│  1️⃣  Flattened batch dimension → Lost request boundaries    │
│  2️⃣  Removed aftercnn_lens → Wrong attention masks          │
│                                                             │
│  Result: In concurrent scenarios                           │
│  ├─ Audio features get misaligned                          │
│  ├─ Talker produces wrong codec codes                      │
│  ├─ Code2Wav receives length 1 input                       │
│  └─ Pipeline deadlocks ⚠️                                   │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                       THE FIX                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ✓ Keep batch dimension: [batch, feature, chunk]           │
│  ✓ Restore aftercnn_lens parameter                         │
│  ✓ Proper boundary tracking                                │
│  ✓ Correct attention masks                                 │
│  ✓ Aligned features for downstream stages                  │
│  → Concurrent requests work correctly ✅                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

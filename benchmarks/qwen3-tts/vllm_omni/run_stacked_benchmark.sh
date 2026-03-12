#!/bin/bash
# Qwen3-TTS Stacked Optimization Benchmark
#
# Runs 4 server configs sequentially (Baseline -> +Batch -> +CUDA Graph ->
# +Async Chunk+Streaming), benchmarking each at multiple concurrency levels.
# Generates pairwise bar charts and summary line plots matching the
# Qwen3-Omni blog figure style.
#
# Usage:
#   bash run_stacked_benchmark.sh
#
# Environment variables:
#   GPU_DEVICE       - GPU index (default: 0)
#   MODEL            - Model path (default: Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
#   NUM_PROMPTS      - Prompts per concurrency level (default: 50)
#   CONCURRENCY      - Space-separated concurrency levels (default: "1 4 10")
#   NUM_WARMUPS      - Warmup requests (default: 3)
#   PORT             - Server port (default: 8000)
#   SKIP_HF          - Set to 1 to skip HF transformers baseline (default: 0)
#   HF_RESULT        - Path to existing HF result JSON (skips HF run if set)

set -eo pipefail

# Activate conda environment (disable nounset temporarily for conda scripts)
eval "$(~/anaconda3/bin/conda shell.bash hook)"
conda activate vllm-omni
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$PROJECT_ROOT"

GPU_DEVICE="${GPU_DEVICE:-0}"
MODEL="${MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}"
NUM_PROMPTS="${NUM_PROMPTS:-50}"
CONCURRENCY="${CONCURRENCY:-1 4 10}"
NUM_WARMUPS="${NUM_WARMUPS:-3}"
PORT="${PORT:-8000}"
SKIP_HF="${SKIP_HF:-0}"
HF_RESULT="${HF_RESULT:-}"
RESULT_DIR="${SCRIPT_DIR}/results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

STAGE_CONFIGS_DIR="vllm_omni/model_executor/stage_configs"

# Config name -> YAML file, whether to stream, description
declare -A CONFIG_YAML CONFIG_STREAM CONFIG_DESC
CONFIG_YAML[baseline]="${STAGE_CONFIGS_DIR}/qwen3_tts_baseline.yaml"
CONFIG_YAML[batch]="${STAGE_CONFIGS_DIR}/qwen3_tts_batch_eager.yaml"
CONFIG_YAML[cuda_graph]="${STAGE_CONFIGS_DIR}/qwen3_tts_batch_cg.yaml"
CONFIG_YAML[async_chunk]="${STAGE_CONFIGS_DIR}/qwen3_tts.yaml"

CONFIG_STREAM[baseline]="--no-stream"
CONFIG_STREAM[batch]="--no-stream"
CONFIG_STREAM[cuda_graph]="--no-stream"
CONFIG_STREAM[async_chunk]=""

CONFIG_DESC[baseline]="Baseline (bs=1, eager, no async)"
CONFIG_DESC[batch]="+ Batch (bs=10, eager, no async)"
CONFIG_DESC[cuda_graph]="+ CUDA Graph (bs=10, CG on Talker, no async)"
CONFIG_DESC[async_chunk]="+ Async Chunk + Streaming (bs=10, CG, async)"

# Order matters - stacked from least to most optimized
CONFIG_ORDER=(baseline batch cuda_graph async_chunk)

mkdir -p "${RESULT_DIR}"

echo "============================================================"
echo " Qwen3-TTS Stacked Optimization Benchmark"
echo "============================================================"
echo " GPU:            ${GPU_DEVICE}"
echo " Model:          ${MODEL}"
echo " Prompts:        ${NUM_PROMPTS}"
echo " Concurrency:    ${CONCURRENCY}"
echo " Port:           ${PORT}"
echo " Results:        ${RESULT_DIR}"
echo " Timestamp:      ${TIMESTAMP}"
echo "============================================================"

SERVER_PID=""
cleanup() {
    if [ -n "$SERVER_PID" ]; then
        echo "Cleaning up server (PID ${SERVER_PID})..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

wait_for_server() {
    local port=$1
    local name=$2
    local max_wait=300
    local elapsed=0
    echo "  Waiting for ${name} server on port ${port}..."
    while ! curl -s "http://localhost:${port}/health" >/dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $elapsed -ge $max_wait ]; then
            echo "  ERROR: ${name} server failed to start within ${max_wait}s"
            exit 1
        fi
    done
    echo "  ${name} server ready (${elapsed}s)"
}

stop_server() {
    if [ -n "$SERVER_PID" ]; then
        echo "  Stopping server (PID ${SERVER_PID})..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
        sleep 5
    fi
}

# Collect result file paths for plotting
declare -A RESULT_FILES

for i in "${!CONFIG_ORDER[@]}"; do
    cfg="${CONFIG_ORDER[$i]}"
    phase=$((i + 1))
    yaml="${CONFIG_YAML[$cfg]}"
    stream_flag="${CONFIG_STREAM[$cfg]}"
    desc="${CONFIG_DESC[$cfg]}"

    echo ""
    echo "============================================================"
    echo " [Phase ${phase}/4] ${desc}"
    echo "   Config: ${yaml}"
    echo "============================================================"

    # Start server
    echo "  Starting server..."
    CUDA_VISIBLE_DEVICES=${GPU_DEVICE} vllm-omni serve "${MODEL}" \
        --stage-configs-path "${yaml}" \
        --host 0.0.0.0 --port "${PORT}" \
        --trust-remote-code --omni \
        > "${RESULT_DIR}/server_${cfg}_${TIMESTAMP}.log" 2>&1 &
    SERVER_PID=$!

    wait_for_server "${PORT}" "${cfg}"

    # Run benchmark
    echo "  Benchmarking ${cfg}..."
    # shellcheck disable=SC2086
    python "${SCRIPT_DIR}/bench_async_chunk.py" \
        --host 127.0.0.1 --port "${PORT}" \
        --config-name "${cfg}" \
        --num-prompts "${NUM_PROMPTS}" \
        --max-concurrency ${CONCURRENCY} \
        --num-warmups "${NUM_WARMUPS}" \
        ${stream_flag} \
        --result-dir "${RESULT_DIR}"

    # Record latest result file
    RESULT_FILES[$cfg]=$(ls -t "${RESULT_DIR}"/bench_${cfg}_*.json 2>/dev/null | head -1)
    echo "  Result: ${RESULT_FILES[$cfg]}"

    stop_server
done

# ---- Collect result paths ----
echo ""
echo "============================================================"
echo " [Phase 5/5] Generating plots"
echo "============================================================"

PLOT_ARGS=""
for cfg in "${CONFIG_ORDER[@]}"; do
    result="${RESULT_FILES[$cfg]:-}"
    if [ -n "$result" ]; then
        PLOT_ARGS="${PLOT_ARGS} --${cfg//_/-} ${result}"
    else
        echo "  WARNING: No result file for ${cfg}, skipping in plots"
    fi
done

# Add HF result if available
if [ -n "$HF_RESULT" ] && [ -f "$HF_RESULT" ]; then
    PLOT_ARGS="${PLOT_ARGS} --hf ${HF_RESULT}"
elif [ "$SKIP_HF" = "0" ]; then
    # Look for existing HF results
    HF_FOUND=$(ls -t "${RESULT_DIR}"/bench_hf_transformers_*.json 2>/dev/null | head -1 || true)
    if [ -n "$HF_FOUND" ]; then
        PLOT_ARGS="${PLOT_ARGS} --hf ${HF_FOUND}"
        echo "  Using existing HF result: ${HF_FOUND}"
    else
        echo "  No HF result found. Run HF benchmark separately and pass via HF_RESULT."
    fi
fi

# Generate plots
# shellcheck disable=SC2086
python "${SCRIPT_DIR}/plot_stacked.py" \
    ${PLOT_ARGS} \
    --output-dir "${RESULT_DIR}" \
    --timestamp "${TIMESTAMP}"

echo ""
echo "============================================================"
echo " Benchmark complete!"
echo " Results: ${RESULT_DIR}/"
echo "============================================================"
echo " Result files:"
for cfg in "${CONFIG_ORDER[@]}"; do
    echo "   ${cfg}: ${RESULT_FILES[$cfg]:-MISSING}"
done
echo ""
echo " Figures saved in: ${RESULT_DIR}/"
echo "============================================================"

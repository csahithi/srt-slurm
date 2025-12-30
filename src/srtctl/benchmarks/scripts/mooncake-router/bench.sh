#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Mooncake Router Benchmark using aiperf
# Tests KV-aware routing vs round-robin using Mooncake conversation trace
# Based on dynamo exemplar for Qwen3-32B
#
# Usage: bench.sh ENDPOINT MODEL_NAME [WORKLOAD] [TTFT_THRESHOLD] [ITL_THRESHOLD]

set -e

ENDPOINT=$1
MODEL_NAME=${2:-"Qwen/Qwen3-32B"}
WORKLOAD=${3:-"conversation"}
TTFT_THRESHOLD=${4:-2000}
ITL_THRESHOLD=${5:-25}

# Setup directories
BASE_DIR="/logs"
TRACE_DIR="${BASE_DIR}/traces"
ARTIFACT_DIR="${BASE_DIR}/artifacts"
mkdir -p "${TRACE_DIR}"
mkdir -p "${ARTIFACT_DIR}"

echo "=============================================="
echo "Mooncake Router Benchmark (aiperf)"
echo "=============================================="
echo "Endpoint: ${ENDPOINT}"
echo "Model: ${MODEL_NAME}"
echo "Workload: ${WORKLOAD}"
echo "TTFT Threshold: ${TTFT_THRESHOLD}ms"
echo "ITL Threshold: ${ITL_THRESHOLD}ms"
echo "=============================================="

# Install aiperf if not present
if ! command -v aiperf &> /dev/null; then
    echo "Installing aiperf..."
    pip install aiperf
fi

# Download Mooncake trace dataset if not already present
declare -A TRACE_URLS
TRACE_URLS["mooncake"]="https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl"
TRACE_URLS["conversation"]="https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/conversation_trace.jsonl"
TRACE_URLS["synthetic"]="https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/synthetic_trace.jsonl"
TRACE_URLS["toolagent"]="https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/toolagent_trace.jsonl"

INPUT_FILE="${TRACE_DIR}/${WORKLOAD}_trace.jsonl"
TRACE_URL="${TRACE_URLS[$WORKLOAD]}"

if [ ! -f "${INPUT_FILE}" ]; then
    echo "Downloading ${WORKLOAD} trace..."
    wget -qO "${INPUT_FILE}" "${TRACE_URL}"
    echo "Downloaded to ${INPUT_FILE}"
fi

# Run small benchmark for warmup
echo "Running small benchmark for warmup..."
aiperf profile \
    -m "${MODEL_NAME}" \
    --url "${ENDPOINT}" \
    --streaming \
    --ui simple \
    --concurrency 10 \
    --request-count 20
echo "Small benchmark for warmup complete"

# Setup artifact directory with model and timestamp
MODEL_BASE_NAME="${MODEL_NAME##*/}"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RUN_ARTIFACT_DIR="${ARTIFACT_DIR}/${MODEL_BASE_NAME}_${WORKLOAD}_${TIMESTAMP}"
mkdir -p "${RUN_ARTIFACT_DIR}"

echo ""
echo "Running aiperf benchmark..."
echo "Input file: ${INPUT_FILE}"
echo "Artifact dir: ${RUN_ARTIFACT_DIR}"
echo ""
echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting benchmark"

# Run aiperf profile exactly as dynamo does
aiperf profile \
    -m "${MODEL_NAME}" \
    --input-file "${INPUT_FILE}" \
    --custom-dataset-type mooncake_trace \
    --fixed-schedule \
    --url "${ENDPOINT}" \
    --streaming \
    --random-seed 42 \
    --ui simple \
    --artifact-dir "${RUN_ARTIFACT_DIR}" \
    --goodput "time_to_first_token:${TTFT_THRESHOLD} inter_token_latency:${ITL_THRESHOLD}"

BENCH_EXIT_CODE=$?

echo ""
echo "$(date '+%Y-%m-%d %H:%M:%S') - Benchmark complete (exit code: ${BENCH_EXIT_CODE})"
echo ""
echo "=============================================="
echo "Mooncake Router Benchmark Complete"
echo "Results saved to: ${RUN_ARTIFACT_DIR}"
echo "=============================================="

# List artifacts
ls -la "${RUN_ARTIFACT_DIR}" 2>/dev/null || true

exit $BENCH_EXIT_CODE

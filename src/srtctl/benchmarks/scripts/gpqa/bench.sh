#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# GPQA accuracy evaluation
# Expects: endpoint [num_examples] [max_tokens] [repeat] [num_threads]

set -e

ENDPOINT=$1
NUM_EXAMPLES=${2:-198}
MAX_TOKENS=${3:-32768}
REPEAT=${4:-8}
NUM_THREADS=${5:-128}

MODEL_NAME="deepseek-ai/DeepSeek-R1"

echo "GPQA Config: endpoint=${ENDPOINT}; num_examples=${NUM_EXAMPLES}; max_tokens=${MAX_TOKENS}; repeat=${REPEAT}; num_threads=${NUM_THREADS}"

# Create results directory
result_dir="/logs/accuracy"
mkdir -p "$result_dir"

# Set OPENAI_API_KEY if not set
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

echo "Running GPQA evaluation..."

cmd=(
    python3 -m sglang.test.run_eval
    --base-url "$ENDPOINT"
    --model "$MODEL_NAME"
    --eval-name gpqa
    --num-examples "$NUM_EXAMPLES"
    --max-tokens "$MAX_TOKENS"
    --repeat "$REPEAT"
    --num-threads "$NUM_THREADS"
)
printf "[CMD] %s\n" "${cmd[*]}"
"${cmd[@]}"

# Copy result file
result_file=$(ls -t /tmp/gpqa_*.json 2>/dev/null | head -n1)
if [ -f "$result_file" ]; then
    cp "$result_file" "$result_dir/"
    echo "Results saved to: $result_dir/$(basename "$result_file")"
else
    echo "Warning: Could not find result file in /tmp"
fi

echo "GPQA evaluation complete"


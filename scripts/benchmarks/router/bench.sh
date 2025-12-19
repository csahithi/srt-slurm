#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Router benchmark using Dynamo's benchmarks/router scripts
# Clones dynamo repo and runs prefix_ratio_benchmark.py

set -e

head_node="localhost"
head_port=8000

n_prefill=$1
n_decode=$2
prefill_gpus=$3
decode_gpus=$4
use_sglang_router=${5:-false}

source /scripts/utils/benchmark_utils.sh

# Determine mode based on worker counts
if [ "$n_prefill" -gt 0 ] && [ "$n_decode" -gt 0 ]; then
    MODE="disagg"
    echo "Running router benchmark in DISAGGREGATED mode"
else
    MODE="agg"
    echo "Running router benchmark in AGGREGATED mode"
fi

echo "Using sglang router: $use_sglang_router"

# Clone dynamo if not present
DYNAMO_DIR="/tmp/dynamo"
if [ ! -d "$DYNAMO_DIR" ]; then
    echo "Cloning dynamo repository..."
    git clone --depth 1 https://github.com/ai-dynamo/dynamo.git "$DYNAMO_DIR"
fi

# Wait for model to be ready
wait_for_model_timeout=3600
wait_for_model_check_interval=5
wait_for_model_report_interval=60
wait_for_model $head_node $head_port $n_prefill $n_decode $wait_for_model_check_interval $wait_for_model_timeout $wait_for_model_report_interval $use_sglang_router

# Install aiperf if needed
pip install aiperf matplotlib 2>/dev/null || true

# Run router benchmark
cd "$DYNAMO_DIR/benchmarks/router"

result_dir="/logs/router-bench"
mkdir -p "$result_dir"

echo "Running prefix ratio benchmark..."
echo "Mode: $MODE"
echo "Results will be saved to: $result_dir"

# Run the prefix ratio benchmark
# Default: prefix ratios 0.5, ISL 14000, OSL 200, 200 requests, concurrency 20
python prefix_ratio_benchmark.py \
    --prefix-ratios 0.1 0.3 0.5 0.7 0.9 \
    --isl 14000 \
    --osl 200 \
    --requests 200 \
    --concurrency 20 \
    --output-dir "$result_dir"

echo "Router benchmark complete. Results in $result_dir"


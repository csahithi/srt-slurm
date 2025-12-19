#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# LongBench-v2 evaluation script using sglang.test.run_eval

head_node="localhost"
head_port=8000
model_name="nvidia/DeepSeek-R1-0528-NVFP4-v2"  # Default model name

# Parse arguments from SLURM job
n_prefill=$1
n_decode=$2
# prefill_gpus and decode_gpus are parsed for argument position consistency with other benchmarks
# shellcheck disable=SC2034
prefill_gpus=$3
# shellcheck disable=SC2034
decode_gpus=$4
num_examples=${5:-}              # Default: all examples
max_tokens=${6:-16384}           # Default: 16384
max_context_length=${7:-128000}  # Default: 128000 (must ensure input + max_tokens <= server context_length)
num_threads=${8:-16}             # Default: 16
# Note: --thinking-mode removed because dynamo frontend doesn't support chat_template_kwargs
categories=${9:-}                # Default: all categories
use_sglang_router=${10:-false}   # Default: false (dynamo frontend)

echo "LongBench-v2 Benchmark Config: num_examples=${num_examples:-all}; max_tokens=${max_tokens}; max_context_length=${max_context_length}; num_threads=${num_threads}; categories=${categories:-all}"

# Source utilities for wait_for_model
source /scripts/utils/benchmark_utils.sh

wait_for_model_timeout=1500 # 25 minutes
wait_for_model_check_interval=5 # check interval -> 5s
wait_for_model_report_interval=60 # wait_for_model report interval -> 60s

wait_for_model $head_node $head_port $n_prefill $n_decode $wait_for_model_check_interval $wait_for_model_timeout $wait_for_model_report_interval $use_sglang_router

# Create results directory
result_dir="/logs/accuracy"
mkdir -p $result_dir

echo "Running LongBench-v2 evaluation..."

# Set OPENAI_API_KEY if not set
if [ -z "$OPENAI_API_KEY" ]; then
    export OPENAI_API_KEY="EMPTY"
fi

# Build the command
# Note: --thinking-mode removed because dynamo frontend doesn't support chat_template_kwargs
cmd="python3 -m sglang.test.run_eval \
    --base-url http://${head_node}:${head_port} \
    --model ${model_name} \
    --eval-name longbench_v2 \
    --max-tokens ${max_tokens} \
    --max-context-length ${max_context_length} \
    --num-threads ${num_threads}"

# Add optional arguments
if [ -n "$num_examples" ]; then
    cmd="$cmd --num-examples ${num_examples}"
fi

if [ -n "$categories" ]; then
    cmd="$cmd --categories ${categories}"
fi

# Run the evaluation
echo "Executing: $cmd"
eval $cmd

# Copy the result file from /tmp to our logs directory
# The result file is named longbench_v2_{model_name}.json
result_file=$(ls -t /tmp/longbench_v2_*.json 2>/dev/null | head -n1)

if [ -f "$result_file" ]; then
    cp "$result_file" "$result_dir/"
    echo "Results saved to: $result_dir/$(basename $result_file)"
else
    echo "Warning: Could not find result file in /tmp"
fi

# Also copy HTML report if available
html_file=$(ls -t /tmp/longbench_v2_*.html 2>/dev/null | head -n1)
if [ -f "$html_file" ]; then
    cp "$html_file" "$result_dir/"
    echo "HTML report saved to: $result_dir/$(basename $html_file)"
fi

echo "LongBench-v2 evaluation complete"


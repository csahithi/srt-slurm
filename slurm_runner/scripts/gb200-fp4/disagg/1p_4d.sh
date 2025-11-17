#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Trying to emulate the trtllm fp4 low latency config 
# config1_tep_tp8_b128_t128
# Uses 1p 4d 
# TRTLLM uses TEP 8 for decode and TEP4 for prefill

# Function to print usage
print_usage() {
    echo "Usage: $0 <mode>"
    echo "  mode: prefill or decode"
    echo ""
    echo "Examples:"
    echo "  $0 prefill"
    echo "  $0 decode"
    exit 1
}

# Check if correct number of arguments provided
if [ $# -ne 1 ]; then
    echo "Error: Expected 1 argument, got $#"
    print_usage
fi

# Parse arguments
mode=$1

# Validate mode argument
if [ "$mode" != "prefill" ] && [ "$mode" != "decode" ]; then
    echo "Error: mode must be 'prefill' or 'decode', got '$mode'"
    print_usage
fi

echo "Mode: $mode"
echo "Command: dynamo"

# Check if required environment variables are set
if [ -z "$HOST_IP_MACHINE" ]; then
    echo "Error: HOST_IP_MACHINE environment variable is not set"
    exit 1
fi

if [ -z "$PORT" ]; then
    echo "Error: PORT environment variable is not set"
    exit 1
fi

if [ -z "$TOTAL_GPUS" ]; then
    echo "Error: TOTAL_GPUS environment variable is not set"
    exit 1
fi

if [ -z "$RANK" ]; then
    echo "Error: RANK environment variable is not set"
    exit 1
fi

if [ -z "$TOTAL_NODES" ]; then
    echo "Error: TOTAL_NODES environment variable is not set"
    exit 1
fi

if [ -z "$USE_INIT_LOCATIONS" ]; then
    echo "Error: USE_INIT_LOCATIONS environment variable is not set"
    exit 1
fi

# Construct command based on mode
if [ "$mode" = "prefill" ]; then
    set -x
    if [[ "${USE_DYNAMO_WHLS,,}" == "true" ]]; then
        python3 -m pip install /configs/ai_dynamo_runtime-0.6.1-cp310-abi3-manylinux_2_28_aarch64.whl
        python3 -m pip install /configs/ai_dynamo-0.6.1-py3-none-any.whl
    fi
    # no expert locations collected for fp4 yet
    command_suffix=""
    if [[ "${USE_INIT_LOCATIONS,,}" == "true" ]]; then command_suffix=" "; fi
    if [[ -n "${DUMP_CONFIG_PATH}" ]]; then command_suffix="${command_suffix} --dump-config-to ${DUMP_CONFIG_PATH}"; fi

    # we have to install pre-release cutedsl for a integer overflow fix
    python3 -m pip install --no-cache-dir --upgrade --pre nvidia-cutlass-dsl

    # set your own cache variables here
    export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800

    # SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN=1 \
    #SGLANG_PER_TOKEN_GROUP_QUANT_8BIT_V2=1 \
    # disable sbo for now

    DYN_SKIP_SGLANG_LOG_FORMATTING=1 \
    SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 \
    SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
    SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
    SGLANG_HACK_SEQ_BOOTSTRAP_ROOM=1 \
    MC_TE_METRIC=true \
    MC_FORCE_MNNVL=1 \
    NCCL_MNNVL_ENABLE=1 \
    NCCL_CUMEM_ENABLE=1 \
    SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True \
    SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
    SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    python3 -m dynamo.sglang \
        --served-model-name deepseek-ai/DeepSeek-R1 \
        --model-path /model/ \
        --disaggregation-mode prefill \
        --decode-log-interval 1000 \
        --max-running-requests 1024 \
        --chunked-prefill-size 8192 \
        --context-length 2200 \
        --enable-flashinfer-allreduce-fusion \
        --enable-symm-mem \
        --disable-radix-cache \
        --disable-shared-experts-fusion \
        --watchdog-timeout 1000000 \
        --moe-runner-backend flashinfer_trtllm \
        --attention-backend trtllm_mla \
        --kv-cache-dtype fp8_e4m3 \
        --trust-remote-code \
        --disable-cuda-graph \
        --load-balance-method round_robin \
        --ep-size "$TOTAL_GPUS" \
        --tp-size "$TOTAL_GPUS" \
        --data-parallel-size 1 \
        --host 0.0.0.0 \
        --stream-interval 20 \
        --log-level debug ${command_suffix}


elif [ "$mode" = "decode" ]; then
    set -x
    if [[ "${USE_DYNAMO_WHLS,,}" == "true" ]]; then
        python3 -m pip install /configs/ai_dynamo_runtime-0.6.1-cp310-abi3-manylinux_2_28_aarch64.whl
        python3 -m pip install /configs/ai_dynamo-0.6.1-py3-none-any.whl
    fi
    # no expert locations collected for fp4 yet
    command_suffix=""
    if [[ "${USE_INIT_LOCATIONS,,}" == "true" ]]; then command_suffix=" "; fi
    if [[ -n "${DUMP_CONFIG_PATH}" ]]; then command_suffix="${command_suffix} --dump-config-to ${DUMP_CONFIG_PATH}"; fi

    # set your own cache variables here
    export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800

    # we have to install pre-release cutedsl for a integer overflow fix
    python3 -m pip install --no-cache-dir --upgrade --pre nvidia-cutlass-dsl

    # SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN=1 \
    # SGLANG_PER_TOKEN_GROUP_QUANT_8BIT_V2=1 \
    # SGLANG_CUTEDSL_MOE_NVFP4_DISPATCH=1 \
    # SGLANG_FLASHINFER_FP4_GEMM_BACKEND=cutlass \
    # flashinfer all reduce fusion?
    # does symm-mem work with TEP?
    # stream interval

    DYN_SKIP_SGLANG_LOG_FORMATTING=1 \
    MC_TE_METRIC=true \
    MC_FORCE_MNNVL=1 \
    NCCL_MNNVL_ENABLE=1 \
    NCCL_CUMEM_ENABLE=1 \
    SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 \
    SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
    SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
    SGLANG_HACK_SEQ_BOOTSTRAP_ROOM=1 \
    SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True \
    PYTHONUNBUFFERED=1 \
    python3 -m dynamo.sglang \
        --served-model-name deepseek-ai/DeepSeek-R1 \
        --model-path /model/ \
        --trust-remote-code \
        --disable-radix-cache \
        --moe-dense-tp-size 1 \
        --max-running-requests 512 \
        --cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256 264 272 280 288 296 304 312 320 328 336 344 352 360 368 376 384 416 448 480 512 544 576 608 640 672 704 736 768 1024 \
        --mem-fraction-static 0.90 \
        --kv-cache-dtype fp8_e4m3 \
        --attention-backend trtllm_mla \
        --stream-interval 20 \
        --enable-symm-mem \
        --moe-runner-backend flashinfer_trtllm \
        --prefill-round-robin-balance \
        --dist-init-addr "$HOST_IP_MACHINE:$PORT" \
        --nnodes "$TOTAL_NODES" \
        --node-rank "$RANK" \
        --host 0.0.0.0 \
        --tp-size "$TOTAL_GPUS" \
        --ep-size "$TOTAL_GPUS" \
        --stream-interval 20 \
        --data-parallel-size 1 ${command_suffix}
fi
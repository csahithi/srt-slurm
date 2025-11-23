#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

wait_for_model() {

    local model_host=$1
    local model_port=$2
    local n_prefill=${3:-1}
    local n_decode=${4:-1}
    local poll=${5:-1}
    local timeout=${6:-600}
    local report_every=${7:-60}

    local health_addr="http://${model_host}:${model_port}/health"
    echo "Polling ${health_addr} every ${poll} seconds to check whether ${n_prefill} prefills and ${n_decode} decodes are alive"

    local start_ts=$(date +%s)
    local report_ts=$(date +%s)

    while :; do
        # Curl timeout - our primary use case here is to launch it at the first node (localhost), so no timeout is needed.
        curl_result=$(curl ${health_addr} 2>/dev/null)
        # Python path - Use of `check_server_health.py` is self-constrained outside of any packaging.
        check_result=$(python3 /scripts/utils/check_server_health.py $n_prefill $n_decode <<< $curl_result)
        if [[ $check_result == *"Model is ready."* ]]; then
            echo $check_result
            return 0
        fi

        time_now=$(date +%s)
        if [[ $((time_now - start_ts)) -ge $timeout ]]; then
            echo "Model did not get healthy in ${timeout} seconds"
            exit 2;
        fi

        if [[ $((time_now - report_ts)) -ge $report_every ]]; then
            echo $check_result
            report_ts=$time_now
        fi

        sleep $poll
    done
}
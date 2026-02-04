#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Hang debugging script: Waits for a configurable time, then collects
# cuda-gdb backtraces from all worker processes.
#
# Usage: collect_backtraces.sh <wait_seconds> <output_dir> <job_id>

set -euo pipefail

WAIT_SECONDS="${1:-600}"
OUTPUT_DIR="${2:-/tmp/backtraces}"
JOB_ID="${3:-unknown}"

echo "[$(date)] Hang debug: Starting backtrace collector"
echo "[$(date)] Will collect backtraces after ${WAIT_SECONDS} seconds"
echo "[$(date)] Output directory: ${OUTPUT_DIR}"
echo "[$(date)] Job ID: ${JOB_ID}"

# Wait for the configured time
echo "[$(date)] Waiting ${WAIT_SECONDS} seconds before collecting backtraces..."
sleep "${WAIT_SECONDS}"

echo "[$(date)] Wait complete. Starting backtrace collection..."

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Find SGLang TP worker processes (scheduler processes that contain the model)
# These are named "sglang::scheduler*" via setproctitle
# The scheduler process contains the TpModelWorker which runs the actual model forward passes
echo "[$(date)] Searching for SGLang TP worker processes (sglang::scheduler*)..."

PIDS=$(pgrep -f "sglang::scheduler" || true)

if [ -z "${PIDS}" ]; then
    echo "[$(date)] No sglang::scheduler processes found, trying fallback search..."
    # Fallback: look for python processes running sglang
    PIDS=$(pgrep -f "python.*sglang" || true)
fi

if [ -z "${PIDS}" ]; then
    echo "[$(date)] WARNING: No TP worker processes found. Nothing to debug."
    ps aux
    exit 0
fi

# Count and list processes
NUM_PROCS=$(echo "${PIDS}" | wc -w | tr -d ' ')
echo "[$(date)] Found ${NUM_PROCS} TP worker process(es): ${PIDS}"

# Show process details for verification
echo "[$(date)] Process details:"
for PID in ${PIDS}; do
    if [ -d "/proc/${PID}" ]; then
        CMDLINE=$(tr '\0' ' ' < "/proc/${PID}/cmdline" 2>/dev/null | head -c 200 || echo "unknown")
        echo "  PID ${PID}: ${CMDLINE}"
    fi
done

# Collect backtrace from each process
for PID in ${PIDS}; do
    echo "[$(date)] Collecting backtrace from PID ${PID}..."

    # Check if process still exists
    if ! kill -0 "${PID}" 2>/dev/null; then
        echo "[$(date)] WARNING: Process ${PID} no longer exists, skipping"
        continue
    fi

    # Get hostname for output filename
    HOSTNAME=$(hostname)
    BACKTRACE_FILE="${OUTPUT_DIR}/backtrace_${HOSTNAME}_${PID}.txt"

    # Collect backtrace using cuda-gdb
    # Use timeout to prevent hanging if gdb gets stuck
    timeout 60 cuda-gdb \
        -batch \
        -ex "set logging redirect on" \
        -ex "set logging file ${BACKTRACE_FILE}" \
        -ex "set logging enabled on" \
        -ex "info cuda kernels" \
        -ex "thread apply all bt" \
        -ex quit \
        -p "${PID}" 2>&1 || {
            echo "[$(date)] ERROR: Failed to collect backtrace from PID ${PID}"
            echo "ERROR: cuda-gdb failed or timed out" >> "${BACKTRACE_FILE}"
        }

    if [ -f "${BACKTRACE_FILE}" ]; then
        echo "[$(date)] Backtrace saved to ${BACKTRACE_FILE}"
    else
        echo "[$(date)] WARNING: Backtrace file not created for PID ${PID}"
    fi
done

echo "[$(date)] Backtrace collection complete"
echo "[$(date)] All backtraces saved to: ${OUTPUT_DIR}"
ls -lh "${OUTPUT_DIR}"

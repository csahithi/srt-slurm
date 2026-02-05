#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Hang debugging script: Waits for a configurable time, then collects
# cuda-gdb backtraces from all worker processes.
#
# Usage: collect_backtraces.sh <wait_seconds> <output_dir> <node> <mode> <index>

set -euo pipefail

WAIT_SECONDS="${1:-600}"
OUTPUT_DIR="${2:-/tmp/backtraces}"
NODE="${3:-$(hostname)}"
MODE="${4:-unknown}"
INDEX="${5:-0}"

# File prefix matching worker log convention: {node}_{mode}_w{index}
FILE_PREFIX="${NODE}_${MODE}_w${INDEX}"

echo "[$(date)] Hang debug: Starting backtrace collector"
echo "[$(date)] Worker: ${FILE_PREFIX}"
echo "[$(date)] Will collect backtraces after ${WAIT_SECONDS} seconds"
echo "[$(date)] Output directory: ${OUTPUT_DIR}"

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
PROC_NUM=0
for PID in ${PIDS}; do
    echo "[$(date)] Collecting backtraces from PID ${PID}..."

    # Check if process still exists
    if ! kill -0 "${PID}" 2>/dev/null; then
        echo "[$(date)] WARNING: Process ${PID} no longer exists, skipping"
        continue
    fi

    # Output filenames matching worker log convention
    # Add process number suffix if multiple processes found
    if [ "${NUM_PROCS}" -gt 1 ]; then
        PYSPY_FILE="${OUTPUT_DIR}/${FILE_PREFIX}_pyspy_p${PROC_NUM}.txt"
        CUDAGDB_FILE="${OUTPUT_DIR}/${FILE_PREFIX}_cudagdb_p${PROC_NUM}.txt"
    else
        PYSPY_FILE="${OUTPUT_DIR}/${FILE_PREFIX}_pyspy.txt"
        CUDAGDB_FILE="${OUTPUT_DIR}/${FILE_PREFIX}_cudagdb.txt"
    fi

    # 1. Collect Python stack trace using py-spy (non-invasive, doesn't pause process)
    echo "[$(date)] Collecting py-spy dump for PID ${PID}..."
    if command -v py-spy &> /dev/null; then
        timeout 30 py-spy dump --pid "${PID}" > "${PYSPY_FILE}" 2>&1 || {
            echo "[$(date)] WARNING: py-spy failed for PID ${PID}"
            echo "ERROR: py-spy failed or timed out" >> "${PYSPY_FILE}"
        }
        if [ -f "${PYSPY_FILE}" ] && [ -s "${PYSPY_FILE}" ]; then
            echo "[$(date)] py-spy output saved to ${PYSPY_FILE}"
        fi
    else
        echo "[$(date)] WARNING: py-spy not found, skipping Python stack trace"
    fi

    # 2. Collect CUDA backtrace using cuda-gdb (more invasive, pauses process briefly)
    echo "[$(date)] Collecting cuda-gdb backtrace for PID ${PID}..."
    if command -v cuda-gdb &> /dev/null; then
        cuda-gdb \
            -batch \
            -ex "set logging redirect on" \
            -ex "set logging file ${CUDAGDB_FILE}" \
            -ex "set logging enabled on" \
            -ex "info cuda kernels" \
            -ex "thread apply all bt" \
            -ex quit \
            -p "${PID}" 2>&1 || {
                echo "[$(date)] WARNING: cuda-gdb failed for PID ${PID}"
                echo "ERROR: cuda-gdb failed or timed out" >> "${CUDAGDB_FILE}"
            }
        if [ -f "${CUDAGDB_FILE}" ] && [ -s "${CUDAGDB_FILE}" ]; then
            echo "[$(date)] cuda-gdb output saved to ${CUDAGDB_FILE}"
        fi
    else
        echo "[$(date)] WARNING: cuda-gdb not found, skipping CUDA backtrace"
    fi

    PROC_NUM=$((PROC_NUM + 1))
done

echo "[$(date)] Backtrace collection complete"
echo "[$(date)] All backtraces saved to: ${OUTPUT_DIR}"
ls -lh "${OUTPUT_DIR}"

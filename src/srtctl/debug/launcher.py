#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launcher for hang debugging background script."""

import logging
import subprocess
from typing import TYPE_CHECKING

from srtctl.core.processes import ManagedProcess

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import DebugConfig

logger = logging.getLogger(__name__)


def launch_hang_debugger(
    debug_config: "DebugConfig",
    runtime: "RuntimeContext",
    worker_nodes: list[str],
) -> list[ManagedProcess]:
    """Launch hang debugging script on all worker nodes.

    The script attaches to the existing worker container on each node using
    --container-name, so it shares the same namespace as the worker processes
    and can access py-spy and cuda-gdb to collect backtraces.

    Args:
        debug_config: Debug configuration
        runtime: Runtime context with paths and node info
        worker_nodes: List of worker node names to monitor

    Returns:
        List of ManagedProcess objects for the debug scripts
    """
    if not debug_config.enabled:
        return []

    logger.info("Launching hang debugger (wait=%ds)", debug_config.wait_seconds)

    # Output directory - use /logs/backtraces inside container since /logs is already mounted
    # The worker container mounts runtime.log_dir -> /logs
    output_dir_host = runtime.log_dir / "backtraces"
    output_dir_container = "/logs/backtraces"

    output_dir_host.mkdir(parents=True, exist_ok=True)
    logger.info("Backtrace output directory: %s (container: %s)", output_dir_host, output_dir_container)

    # The debug script is in benchmarks/scripts/debug/, which is mounted at /srtctl-benchmarks/
    container_script_path = "/srtctl-benchmarks/debug/collect_backtraces.sh"

    processes = []

    # Launch script on each worker node, attaching to the existing worker container
    for node in worker_nodes:
        log_file = runtime.log_dir / f"hang_debug_{node}.out"

        # Use the same container name as the worker to attach to it
        # This must match the name used in worker_stage.py
        container_name = f"sglang_{runtime.job_id}_{node}"

        logger.info("Starting hang debugger on %s (attaching to container %s)", node, container_name)

        # Build srun command to attach to the existing worker container
        # No additional mounts needed - we use paths already mounted by the worker
        cmd = [
            "srun",
            "--overlap",
            "--nodes=1",
            f"--nodelist={node}",
            "--ntasks=1",
            f"--output={log_file}",
            "--open-mode=append",
            # Attach to the existing worker container by name
            f"--container-name={container_name}",
            # Run the script inside the existing container
            "bash",
            container_script_path,
            str(debug_config.wait_seconds),
            output_dir_container,
            str(runtime.job_id),
        ]

        logger.debug("Hang debugger command: %s", " ".join(cmd))

        # Launch in background
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        managed = ManagedProcess(
            name=f"hang_debug_{node}",
            popen=proc,
            log_file=log_file,
            node=node,
            critical=False,  # Non-critical - shouldn't stop the job if it fails
        )
        processes.append(managed)

    logger.info("Launched hang debugger on %d nodes", len(processes))
    return processes

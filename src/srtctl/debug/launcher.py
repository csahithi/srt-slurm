#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launcher for hang debugging background script."""

import logging
import subprocess
from pathlib import Path
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

    # Determine output directory
    output_dir = Path(debug_config.output_dir) if debug_config.output_dir else runtime.log_dir / "backtraces"

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Backtrace output directory: %s", output_dir)

    # Get script path
    script_path = Path(__file__).parent / "scripts" / "collect_backtraces.sh"
    if not script_path.exists():
        logger.error("Hang debug script not found: %s", script_path)
        return []

    processes = []

    # Launch script on each worker node
    for node in worker_nodes:
        log_file = runtime.log_dir / f"hang_debug_{node}.out"
        logger.info("Starting hang debugger on %s (log: %s)", node, log_file)

        # Build srun command to run script on specific node
        cmd = [
            "srun",
            "--nodes=1",
            f"--nodelist={node}",
            "--ntasks=1",
            f"--output={log_file}",
            "--open-mode=append",
            str(script_path),
            str(debug_config.wait_seconds),
            str(output_dir),
            str(runtime.job_id),
        ]

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

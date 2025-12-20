# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Utility functions for the orchestrator.

This module provides:
- start_srun_process(): Typed wrapper around subprocess srun calls
- wait_for_port(): Poll TCP port availability
- wait_for_health(): HTTP health check with worker count validation
- get_hostname_ip(): Resolve node hostname to IP
"""

import logging
import shlex
import socket
import subprocess
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import requests

from srtctl.scripts import get_node_ip  # noqa: E402

logger = logging.getLogger(__name__)


def start_srun_process(
    command: list[str],
    *,
    nodes: int = 1,
    ntasks: int = 1,
    cpus_per_task: int | None = None,
    nodelist: Sequence[str] | None = None,
    output: str | None = None,
    container_image: str | None = None,
    container_mounts: dict[Path, Path] | None = None,
    env_to_pass_through: list[str] | None = None,
    env_to_set: dict[str, str] | None = None,
    bash_preamble: str | None = None,
    srun_options: dict[str, str] | None = None,
    overlap: bool = True,
    use_bash_wrapper: bool = True,
) -> subprocess.Popen:
    """Start a process via srun with container support.

    This is the central function for launching all srun processes.
    It handles container mounts, environment variables, and output redirection.

    Args:
        command: Command to run as list of strings
        nodes: Number of nodes (default: 1)
        ntasks: Number of tasks (default: 1)
        cpus_per_task: CPUs per task (optional)
        nodelist: Specific nodes to run on (optional)
        output: Output file path (optional)
        container_image: Container image path (optional)
        container_mounts: Dict of host_path -> container_path mounts
        env_to_pass_through: Environment variable names to pass through
        env_to_set: Environment variables to set (name -> value)
        bash_preamble: Bash commands to run before the main command
        srun_options: Additional srun options as dict
        overlap: Use --overlap flag (default: True)
        use_bash_wrapper: Wrap command in bash -c (default: True)

    Returns:
        subprocess.Popen object for the srun process

    Example:
        proc = start_srun_process(
            command=["python3", "-m", "dynamo.sglang", "--model-path", "/model"],
            nodelist=["node1"],
            container_image="/containers/sglang.sqsh",
            container_mounts={Path("/models/llama"): Path("/model")},
            env_to_set={"NATS_SERVER": "nats://node1:4222"},
        )
    """
    srun_cmd = ["srun"]

    # Basic options
    if overlap:
        srun_cmd.append("--overlap")

    srun_cmd.extend(["--nodes", str(nodes)])
    srun_cmd.extend(["--ntasks", str(ntasks)])

    if cpus_per_task:
        srun_cmd.extend(["--cpus-per-task", str(cpus_per_task)])

    if nodelist:
        srun_cmd.extend(["--nodelist", ",".join(nodelist)])

    if output:
        srun_cmd.extend(["--output", output])

    # Container options
    if container_image:
        srun_cmd.extend(["--container-image", str(container_image)])
        srun_cmd.append("--no-container-entrypoint")
        srun_cmd.append("--no-container-mount-home")

        if container_mounts:
            mount_str = ",".join(
                f"{host}:{container}"
                for host, container in container_mounts.items()
            )
            srun_cmd.extend(["--container-mounts", mount_str])

    # Additional srun options
    if srun_options:
        for key, value in srun_options.items():
            if value:
                srun_cmd.extend([f"--{key}", value])
            else:
                srun_cmd.append(f"--{key}")

    # Build the actual command to run
    if use_bash_wrapper:
        # Build bash command with environment setup
        bash_parts = []

        # Add preamble if provided
        if bash_preamble:
            bash_parts.append(bash_preamble)

        # Export environment variables
        if env_to_set:
            for name, value in env_to_set.items():
                bash_parts.append(f"export {name}={shlex.quote(value)}")

        # Add the main command
        bash_parts.append(shlex.join(command))

        # Join with && for sequential execution
        bash_command = " && ".join(bash_parts)
        srun_cmd.extend(["bash", "-c", bash_command])
    else:
        srun_cmd.extend(command)

    logger.debug("Starting srun: %s", shlex.join(srun_cmd))

    # Start the process
    proc = subprocess.Popen(
        srun_cmd,
        stdout=subprocess.PIPE if not output else None,
        stderr=subprocess.STDOUT if not output else None,
        env=None,  # Inherit environment
    )

    return proc


def wait_for_port(
    host: str,
    port: int,
    timeout: float = 60.0,
    interval: float = 1.0,
) -> bool:
    """Wait for a TCP port to become available.

    Args:
        host: Hostname or IP address
        port: Port number
        timeout: Maximum time to wait in seconds
        interval: Time between checks in seconds

    Returns:
        True if port became available, False if timeout
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except (TimeoutError, ConnectionRefusedError, OSError):
            time.sleep(interval)

    return False


def wait_for_health(
    host: str,
    port: int,
    max_attempts: int = 60,
    interval: float = 10.0,
    expected_workers: int | None = None,
    stop_event: threading.Event | None = None,
) -> bool:
    """Wait for HTTP health endpoint to return healthy status.

    Checks /health endpoint and optionally /v1/models for worker readiness.

    Args:
        host: Hostname or IP address
        port: HTTP port
        max_attempts: Maximum number of attempts
        interval: Time between attempts in seconds
        expected_workers: Expected number of workers (checks /v1/models)
        stop_event: Optional threading.Event to abort waiting

    Returns:
        True if healthy, False if timeout or aborted
    """
    health_url = f"http://{host}:{port}/health"
    models_url = f"http://{host}:{port}/v1/models"

    for attempt in range(max_attempts):
        if stop_event and stop_event.is_set():
            logger.warning("Wait aborted by stop event")
            return False

        try:
            # Check health endpoint
            response = requests.get(health_url, timeout=5.0)
            if response.status_code != 200:
                logger.debug(
                    "Health check failed (attempt %d/%d): status %d",
                    attempt + 1,
                    max_attempts,
                    response.status_code,
                )
                time.sleep(interval)
                continue

            # If expected_workers specified, check /v1/models
            if expected_workers is not None:
                try:
                    models_response = requests.get(models_url, timeout=5.0)
                    if models_response.status_code == 200:
                        data = models_response.json()
                        # Check if we have the expected number of workers
                        # The response format depends on the backend
                        models = data.get("data", [])
                        if len(models) > 0:
                            logger.info(
                                "Health check passed: %d models available",
                                len(models),
                            )
                            return True
                except Exception as e:
                    logger.debug("Models check failed: %s", e)
                    time.sleep(interval)
                    continue
            else:
                logger.info("Health check passed")
                return True

        except requests.exceptions.RequestException as e:
            logger.debug(
                "Health check failed (attempt %d/%d): %s",
                attempt + 1,
                max_attempts,
                e,
            )

        time.sleep(interval)

    logger.error("Health check failed after %d attempts", max_attempts)
    return False


def wait_for_etcd(
    etcd_url: str,
    max_retries: int = 60,
    interval: float = 2.0,
) -> bool:
    """Wait for etcd to be ready.

    Args:
        etcd_url: Base URL of etcd (e.g., http://node1:2379)
        max_retries: Maximum number of retries
        interval: Time between retries in seconds

    Returns:
        True if etcd is ready, False if timeout
    """
    health_url = f"{etcd_url}/health"

    for attempt in range(max_retries):
        try:
            response = requests.get(health_url, timeout=5.0)
            if response.status_code == 200:
                logger.info("etcd is ready")
                return True
        except requests.exceptions.RequestException:
            pass

        logger.debug(
            "etcd not ready (attempt %d/%d), retrying...",
            attempt + 1,
            max_retries,
        )
        time.sleep(interval)

    logger.error("etcd not ready after %d attempts", max_retries)
    return False


def get_container_mounts_str(mounts: dict[Path, Path]) -> str:
    """Convert container mounts dict to comma-separated string.

    Args:
        mounts: Dict mapping host paths to container paths

    Returns:
        Comma-separated string for --container-mounts
    """
    return ",".join(f"{host}:{container}" for host, container in mounts.items())


def run_command(
    command: str,
    background: bool = False,
    stdout=None,
    stderr=None,
) -> subprocess.Popen | int:
    """Run a shell command.

    Args:
        command: Command string to run
        background: If True, return Popen object; if False, wait and return exit code
        stdout: Optional stdout file handle
        stderr: Optional stderr file handle

    Returns:
        Popen object if background=True, exit code if background=False
    """
    logger.debug("Running command: %s", command)

    if background:
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=stdout or subprocess.DEVNULL,
            stderr=stderr or subprocess.DEVNULL,
        )
        return proc
    else:
        result = subprocess.run(command, shell=True)
        return result.returncode


def get_node_ips(
    nodes: list[str],
    slurm_job_id: str | None = None,
    network_interface: str | None = None,
) -> dict[str, str]:
    """Get IP addresses for multiple SLURM nodes.

    Args:
        nodes: List of node hostnames
        slurm_job_id: SLURM job ID for srun context
        network_interface: Specific network interface to use

    Returns:
        Dict mapping node hostname to IP address
    """
    ips = {}
    for node in nodes:
        ip = get_node_ip(node, slurm_job_id, network_interface)
        if ip:
            ips[node] = ip
        else:
            logger.warning("Could not resolve IP for node %s", node)
    return ips


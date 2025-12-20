# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Bash script wrappers for battle-tested SLURM utilities.

These scripts are copied from scripts/utils/ and are called directly
to avoid reimplementing complex logic in Python.
"""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Path to the scripts directory
SCRIPTS_DIR = Path(__file__).parent


def get_node_ip(
    node: str,
    slurm_job_id: str | None = None,
    network_interface: str | None = None,
    timeout: float = 30.0,
) -> str | None:
    """Get IP address for a SLURM node using the battle-tested bash function.

    Uses scripts/slurm_utils.sh::get_node_ip which tries multiple methods:
    1. Specific network interface (if provided)
    2. hostname -I (gets first non-loopback IP)
    3. ip route get 8.8.8.8 (finds default source IP)

    Args:
        node: Hostname of the node to query
        slurm_job_id: SLURM job ID for srun context
        network_interface: Specific network interface to use
        timeout: Command timeout in seconds

    Returns:
        IP address string, or None if resolution failed
    """
    slurm_utils = SCRIPTS_DIR / "slurm_utils.sh"

    # Build bash command that sources the script and calls the function
    bash_cmd = f"""
        source "{slurm_utils}"
        get_node_ip "{node}" "{slurm_job_id or ''}" "{network_interface or ''}"
    """

    try:
        result = subprocess.run(
            ["bash", "-c", bash_cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode == 0 and result.stdout.strip():
            ip = result.stdout.strip()
            logger.debug("Resolved IP for %s: %s", node, ip)
            return ip
        else:
            logger.error(
                "Failed to get IP for node %s (exit %d): %s",
                node,
                result.returncode,
                result.stderr,
            )
            return None

    except subprocess.TimeoutExpired:
        logger.error("Timeout getting IP for node %s", node)
        return None
    except Exception as e:
        logger.error("Error getting IP for node %s: %s", node, e)
        return None


def get_local_ip(network_interface: str | None = None) -> str:
    """Get local IP address using the same methods as get_node_ip.

    This runs locally (no srun) and tries:
    1. Specific network interface (if provided)
    2. hostname -I (gets first non-loopback IP)
    3. ip route get 8.8.8.8 (finds default source IP)

    Args:
        network_interface: Specific network interface to use

    Returns:
        IP address string, or "127.0.0.1" if all methods fail
    """
    # Method 1: Specific interface
    if network_interface:
        try:
            result = subprocess.run(
                ["bash", "-c", f"ip addr show {network_interface} 2>/dev/null | grep 'inet ' | awk '{{print $2}}' | cut -d'/' -f1"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass

    # Method 2: hostname -I
    try:
        result = subprocess.run(
            ["hostname", "-I"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            ip = result.stdout.strip().split()[0]
            if ip and not ip.startswith("127."):
                return ip
    except Exception:
        pass

    # Method 3: ip route get 8.8.8.8
    try:
        result = subprocess.run(
            ["bash", "-c", "ip route get 8.8.8.8 2>/dev/null | awk -F'src ' 'NR==1{split($2,a,\" \");print a[1]}'"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            ip = result.stdout.strip()
            if ip and not ip.startswith("127."):
                return ip
    except Exception:
        pass

    logger.warning("Could not determine local IP, using 127.0.0.1")
    return "127.0.0.1"


def wait_for_model(
    model_host: str,
    model_port: int,
    n_prefill: int = 1,
    n_decode: int = 1,
    poll: int = 1,
    timeout: int = 600,
    report_every: int = 60,
    use_sglang_router: bool = False,
) -> bool:
    """Wait for model to be ready using the battle-tested bash function.

    Uses scripts/benchmark_utils.sh::wait_for_model

    Args:
        model_host: Model server hostname
        model_port: Model server port
        n_prefill: Expected number of prefill workers
        n_decode: Expected number of decode workers
        poll: Poll interval in seconds
        timeout: Maximum wait time in seconds
        report_every: Report progress every N seconds
        use_sglang_router: Whether using sglang router (uses /workers endpoint)

    Returns:
        True if model is ready, False if timeout
    """
    benchmark_utils = SCRIPTS_DIR / "benchmark_utils.sh"
    router_flag = "true" if use_sglang_router else "false"

    bash_cmd = f"""
        source "{benchmark_utils}"
        wait_for_model "{model_host}" "{model_port}" "{n_prefill}" "{n_decode}" "{poll}" "{timeout}" "{report_every}" "{router_flag}"
    """

    try:
        result = subprocess.run(
            ["bash", "-c", bash_cmd],
            capture_output=True,
            text=True,
            timeout=timeout + 30,  # Extra buffer
        )

        if result.returncode == 0:
            logger.info("Model is ready: %s", result.stdout.strip().split('\n')[-1])
            return True
        else:
            logger.error("Model failed to become ready: %s", result.stderr or result.stdout)
            return False

    except subprocess.TimeoutExpired:
        logger.error("Timeout waiting for model")
        return False
    except Exception as e:
        logger.error("Error waiting for model: %s", e)
        return False


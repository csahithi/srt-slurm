# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Infrastructure setup for NATS, ETCD, Nginx, and frontend workers."""

import logging
import os
from typing import Any

from .command import install_dynamo_wheels
from .environment import ETCD_CLIENT_PORT, ETCD_LISTEN_ADDR, ETCD_PEER_PORT
from .utils import run_command, wait_for_etcd


def _build_extra_args(args_dict: dict[str, Any] | None) -> str:
    """Convert args dict to CLI flags string.
    
    Args:
        args_dict: Dict of arg names to values. Boolean True = flag with no value.
        
    Returns:
        CLI args string like "--kv-overlap-score-weight 1 --no-kv-events"
    """
    if not args_dict:
        return ""
    
    parts = []
    for key, value in args_dict.items():
        if value is True:
            parts.append(f"--{key}")
        elif value is not False and value is not None:
            parts.append(f"--{key} {value}")
    return " ".join(parts)


def setup_head_prefill_node(prefill_host_ip: str) -> None:
    """Setup NATS and ETCD on the prefill host node."""
    logging.info(f"Starting nats server on node {prefill_host_ip}")
    nats_cmd = "/configs/nats-server -js"
    nats_process = run_command(nats_cmd, background=True)
    if not nats_process:
        raise RuntimeError("Failed to start nats-server")

    logging.info(f"Starting etcd server on node {prefill_host_ip}")
    etcd_cmd = (
        f"/configs/etcd --listen-client-urls {ETCD_LISTEN_ADDR}:{ETCD_CLIENT_PORT} "
        f"--advertise-client-urls {ETCD_LISTEN_ADDR}:{ETCD_CLIENT_PORT} "
        f"--listen-peer-urls {ETCD_LISTEN_ADDR}:{ETCD_PEER_PORT} "
        f"--initial-cluster default=http://{prefill_host_ip}:{ETCD_PEER_PORT}"
    )

    etcd_process = run_command(etcd_cmd, background=True)
    if not etcd_process:
        raise RuntimeError("Failed to start etcd")


def setup_nginx_worker(nginx_config: str) -> int:
    """Setup nginx load balancer"""
    logging.info("Setting up nginx load balancer")

    if not nginx_config or not os.path.exists(nginx_config):
        raise ValueError(f"Nginx config file not found: {nginx_config}")

    nginx_cmd = f"apt-get update && apt-get install -y nginx && nginx -c {nginx_config} && sleep 86400"
    return run_command(nginx_cmd)


def setup_frontend_worker(
    worker_idx: int,
    master_ip: str,
    gpu_type: str,
    extra_args: dict[str, Any] | None = None,
) -> int:
    """Setup a frontend worker.
    
    Args:
        worker_idx: Index of this frontend (0 = primary with NATS/ETCD)
        master_ip: IP of the master node (for NATS/ETCD)
        gpu_type: GPU type string
        extra_args: Extra CLI args to pass to dynamo.frontend
    """
    logging.info(f"Setting up frontend worker {worker_idx}")

    # First frontend (worker_idx 0) also sets up NATS/ETCD
    if worker_idx == 0:
        setup_head_prefill_node(master_ip)
        if not wait_for_etcd(f"http://{master_ip}:{ETCD_CLIENT_PORT}"):
            raise RuntimeError("Failed to connect to etcd")
    else:
        logging.info(f"Setting up additional frontend worker {worker_idx}")
        if not wait_for_etcd(f"http://{master_ip}:{ETCD_CLIENT_PORT}"):
            raise RuntimeError("Failed to connect to etcd")

    # Install dynamo from PyPI
    install_dynamo_wheels(gpu_type)

    # Run frontend with extra args
    extra_args_str = _build_extra_args(extra_args)
    frontend_cmd = f"python3 -m dynamo.frontend --http-port=8000 {extra_args_str}".strip()
    logging.info(f"Frontend command: {frontend_cmd}")
    return run_command(frontend_cmd)

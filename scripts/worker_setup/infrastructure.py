# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Infrastructure setup for NATS, ETCD, Nginx, and frontend workers."""

import logging
import os

from .command import install_dynamo_wheels
from .environment import ETCD_CLIENT_PORT, ETCD_LISTEN_ADDR, ETCD_PEER_PORT
from .utils import run_command, wait_for_etcd


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


def setup_frontend_worker(worker_idx: int, master_ip: str, gpu_type: str) -> int:
    """Setup a frontend worker"""
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

    # Run frontend
    frontend_cmd = "python3 -m dynamo.frontend --http-port=8000"
    return run_command(frontend_cmd)


def setup_sglang_router(
    prefill_ips: list[str],
    decode_ips: list[str],
    prefill_bootstrap_port: int,
    port: int = 8000,
) -> int:
    """Setup sglang_router instead of dynamo.frontend

    Args:
        prefill_ips: List of prefill worker leader IP addresses
        decode_ips: List of decode worker leader IP addresses
        prefill_bootstrap_port: Bootstrap port for prefill workers
        port: Port for router to listen on (default: 8000)

    Returns:
        Exit code from router command
    """
    logging.info(f"Setting up sglang_router on port {port}")

    # Log all prefill endpoints
    for i, ip in enumerate(prefill_ips):
        logging.info(f"Prefill worker {i}: http://{ip}:30000 (bootstrap: {prefill_bootstrap_port})")

    # Log all decode endpoints
    for i, ip in enumerate(decode_ips):
        logging.info(f"Decode worker {i}: http://{ip}:30001")

    # Build router command with multiple --prefill and --decode arguments
    router_cmd_parts = [
        "python3 -m sglang_router.launch_router",
        "--pd-disaggregation",
        "--host 0.0.0.0",
        f"--port {port}",
    ]

    # Add all prefill endpoints
    for ip in prefill_ips:
        router_cmd_parts.append(f"--prefill http://{ip}:30000 {prefill_bootstrap_port}")

    # Add all decode endpoints
    for ip in decode_ips:
        router_cmd_parts.append(f"--decode http://{ip}:30001")

    router_cmd = " ".join(router_cmd_parts)

    return run_command(router_cmd)

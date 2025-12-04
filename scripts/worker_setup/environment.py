# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment configuration constants and setup."""

import logging
import os

# Network configurations
ETCD_CLIENT_PORT = 2379
ETCD_PEER_PORT = 2380
NATS_PORT = 4222
DIST_INIT_PORT = 29500
ETCD_LISTEN_ADDR = "http://0.0.0.0"


def setup_env(master_ip: str):
    """Setup NATS and ETCD environment variables."""
    nats_server = f"nats://{master_ip}:{NATS_PORT}"
    etcd_endpoints = f"http://{master_ip}:{ETCD_CLIENT_PORT}"

    os.environ["NATS_SERVER"] = nats_server
    os.environ["ETCD_ENDPOINTS"] = etcd_endpoints

    # TEMP
    os.environ["PATH"] = os.environ.get("PATH", "") + ":/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin"
    cicc_path = os.popen("which cicc").read().strip()
    nvcc_version = os.popen("nvcc --version").read().strip()
    logging.info(f"which cicc: {cicc_path}")
    logging.info(f"nvcc --version:\n{nvcc_version}")

    logging.info(f"set NATS_SERVER: {nats_server}")
    logging.info(f"set ETCD_ENDPOINTS: {etcd_endpoints}")

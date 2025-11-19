# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment variable setup for workers."""

import logging
import os

# Network configurations
ETCD_CLIENT_PORT = 2379
ETCD_PEER_PORT = 2380
NATS_PORT = 4222
DIST_INIT_PORT = 29500
ETCD_LISTEN_ADDR = "http://0.0.0.0"


def setup_env_vars_for_gpu_script(
    host_ip: str,
    local_rank: int,
    total_gpus: int,
    total_nodes: int,
    port: int = DIST_INIT_PORT,
    use_init_locations: bool = True,
    dump_config_path: str | None = None,
    use_dynamo_whls: bool = False,
    sglang_torch_profiler: bool = False,
    worker_type: str = "aggregated",
):
    """Setup environment variables required by GPU scripts"""
    os.environ["HOST_IP_MACHINE"] = host_ip
    os.environ["PORT"] = str(port)
    os.environ["TOTAL_GPUS"] = str(total_gpus)
    os.environ["RANK"] = str(local_rank)
    os.environ["TOTAL_NODES"] = str(total_nodes)
    os.environ["USE_INIT_LOCATIONS"] = str(use_init_locations)
    os.environ["USE_DYNAMO_WHLS"] = str(use_dynamo_whls)
    os.environ["USE_SGLANG_LAUNCH_SERVER"] = str(sglang_torch_profiler)
    if sglang_torch_profiler:
        # Set profiler directory with worker-type-specific subdirectory
        os.environ["SGLANG_TORCH_PROFILER_DIR"] = f"/logs/profiles/{worker_type}"
    if dump_config_path:
        os.environ["DUMP_CONFIG_PATH"] = dump_config_path
    else:
        os.environ.pop("DUMP_CONFIG_PATH", None)

    logging.info(f"Set HOST_IP: {host_ip}")
    logging.info(f"Set PORT: {port}")
    logging.info(f"Set TOTAL_GPUS: {total_gpus}")
    logging.info(f"Set RANK: {local_rank}")
    logging.info(f"Set TOTAL_NODES: {total_nodes}")
    logging.info(f"Set USE_INIT_LOCATIONS: {use_init_locations}")
    logging.info(f"Set USE_DYNAMO_WHLS: {use_dynamo_whls}")
    logging.info(f"Set USE_SGLANG_LAUNCH_SERVER: {sglang_torch_profiler}")
    if sglang_torch_profiler:
        logging.info(f"Set SGLANG_TORCH_PROFILER_DIR: /logs/profiles/{worker_type}")
    if dump_config_path:
        logging.info(f"Set DUMP_CONFIG_PATH: {dump_config_path}")


def setup_env(master_ip: str):
    """Setup NATS and ETCD environment variables."""
    nats_server = f"nats://{master_ip}:{NATS_PORT}"
    etcd_endpoints = f"http://{master_ip}:{ETCD_CLIENT_PORT}"

    os.environ["NATS_SERVER"] = nats_server
    os.environ["ETCD_ENDPOINTS"] = etcd_endpoints

    logging.info(f"set NATS_SERVER: {nats_server}")
    logging.info(f"set ETCD_ENDPOINTS: {etcd_endpoints}")

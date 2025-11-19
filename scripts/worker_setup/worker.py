# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker setup functions for prefill, decode, and aggregated workers."""

import logging

from .command import get_gpu_command, install_dynamo_wheels
from .environment import ETCD_CLIENT_PORT, setup_env_vars_for_gpu_script
from .infrastructure import setup_head_prefill_node
from .utils import run_command, wait_for_etcd


def setup_prefill_worker(
    worker_idx: int,
    local_rank: int,
    leader_ip: str,
    master_ip: str,
    nodes_per_worker: int,
    gpus_per_node: int,
    gpu_type: str,
    script_variant: str,
    multiple_frontends_enabled: bool = False,
    use_init_locations: bool = True,
    dump_config_path: str | None = None,
    use_dynamo_whls: bool = False,
    sglang_torch_profiler: bool = False,
    sglang_config_path: str | None = None,
) -> int:
    """
    Setup the prefill worker.
    """
    total_gpus = nodes_per_worker * gpus_per_node
    # Only setup infrastructure in traditional mode (not multiple frontends)
    if not multiple_frontends_enabled and worker_idx == 0 and local_rank == 0:
        setup_head_prefill_node(master_ip, use_dynamo_whls)
    else:
        logging.info(f"Setting up prefill worker {worker_idx}, local rank {local_rank}")
        if not wait_for_etcd(f"http://{master_ip}:{ETCD_CLIENT_PORT}"):
            raise RuntimeError("Failed to connect to etcd")

    # Setup environment variables for GPU script - use leader_ip as dist-init-addr
    setup_env_vars_for_gpu_script(
        leader_ip,
        local_rank,
        total_gpus,
        nodes_per_worker,
        use_init_locations=use_init_locations,
        dump_config_path=dump_config_path,
        use_dynamo_whls=use_dynamo_whls,
        sglang_torch_profiler=sglang_torch_profiler,
        worker_type="prefill",
    )

    # Install dynamo wheels if needed
    install_dynamo_wheels(gpu_type)

    # Build command from YAML config
    cmd_to_run = get_gpu_command("prefill", sglang_config_path)
    return run_command(cmd_to_run)


def setup_decode_worker(
    worker_idx: int,
    local_rank: int,
    leader_ip: str,
    master_ip: str,
    nodes_per_worker: int,
    gpus_per_node: int,
    gpu_type: str,
    script_variant: str,
    use_init_locations: bool = True,
    dump_config_path: str | None = None,
    use_dynamo_whls: bool = False,
    sglang_torch_profiler: bool = False,
    sglang_config_path: str | None = None,
) -> int:
    """
    Setup the decode worker.
    """
    total_gpus = nodes_per_worker * gpus_per_node
    logging.info(f"Setting up decode worker {worker_idx}, local rank {local_rank}")

    if not wait_for_etcd(f"http://{master_ip}:{ETCD_CLIENT_PORT}"):
        raise RuntimeError("Failed to connect to etcd")

    # Setup environment variables for GPU script - use leader_ip as dist-init-addr
    setup_env_vars_for_gpu_script(
        leader_ip,
        local_rank,
        total_gpus,
        nodes_per_worker,
        use_init_locations=use_init_locations,
        dump_config_path=dump_config_path,
        use_dynamo_whls=use_dynamo_whls,
        sglang_torch_profiler=sglang_torch_profiler,
        worker_type="decode",
    )

    # Install dynamo wheels if needed
    install_dynamo_wheels(gpu_type)

    # Build command from YAML config
    cmd_to_run = get_gpu_command("decode", sglang_config_path)
    return run_command(cmd_to_run)


def setup_aggregated_worker(
    worker_idx: int,
    local_rank: int,
    leader_ip: str,
    master_ip: str,
    nodes_per_worker: int,
    gpus_per_node: int,
    gpu_type: str,
    script_variant: str,
    multiple_frontends_enabled: bool = False,
    dump_config_path: str | None = None,
    use_dynamo_whls: bool = False,
    sglang_torch_profiler: bool = False,
    sglang_config_path: str | None = None,
) -> int:
    """
    Setup the aggregated worker.
    """
    total_gpus = nodes_per_worker * gpus_per_node
    # Only setup infrastructure in traditional mode (not multiple frontends) on first worker, first node
    if not multiple_frontends_enabled and worker_idx == 0 and local_rank == 0:
        setup_head_prefill_node(master_ip, use_dynamo_whls)
    else:
        logging.info(f"Setting up aggregated worker {worker_idx}, local rank {local_rank}")
        if not wait_for_etcd(f"http://{master_ip}:{ETCD_CLIENT_PORT}"):
            raise RuntimeError("Failed to connect to etcd")

    # Setup environment variables for GPU script - use leader_ip as dist-init-addr
    # Aggregated mode doesn't use init locations
    setup_env_vars_for_gpu_script(
        leader_ip,
        local_rank,
        total_gpus,
        nodes_per_worker,
        use_init_locations=False,
        dump_config_path=dump_config_path,
        use_dynamo_whls=use_dynamo_whls,
        sglang_torch_profiler=sglang_torch_profiler,
        worker_type="aggregated",
    )

    # Install dynamo wheels if needed
    install_dynamo_wheels(gpu_type)

    # Build command from YAML config
    cmd_to_run = get_gpu_command("aggregated", sglang_config_path)
    return run_command(cmd_to_run)

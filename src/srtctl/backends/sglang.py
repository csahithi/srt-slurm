# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SGLang backend helper functions.

The main SGLangBackendConfig is defined in srtctl.core.schema and implements
the BackendProtocol. This module provides standalone helper functions for
building SGLang commands.
"""

from pathlib import Path
from typing import Any, Literal

WorkerMode = Literal["prefill", "decode", "agg"]


def build_sglang_command(
    mode: WorkerMode,
    config: dict[str, Any],
    model_path: Path,
    served_model_name: str,
    leader_ip: str,
    dist_init_port: int = 29500,
    num_nodes: int = 1,
    node_rank: int = 0,
    use_sglang_router: bool = False,
    dump_config_path: Path | None = None,
) -> list[str]:
    """Build an SGLang command from configuration.

    This is a standalone function for use in templates or scripts.

    Args:
        mode: Worker mode (prefill, decode, agg)
        config: SGLang config dict for this mode
        model_path: Path to model
        served_model_name: Model name to serve
        leader_ip: IP of the leader node
        dist_init_port: Port for distributed init
        num_nodes: Total nodes in this endpoint
        node_rank: This node's rank
        use_sglang_router: Use sglang.launch_server instead of dynamo.sglang
        dump_config_path: Optional path to dump config

    Returns:
        Command as list of strings
    """
    python_module = "sglang.launch_server" if use_sglang_router else "dynamo.sglang"

    cmd = [
        "python3", "-m", python_module,
        "--model-path", str(model_path),
        "--served-model-name", served_model_name,
        "--host", "0.0.0.0",
    ]

    # Add disaggregation mode
    if mode != "agg" and not use_sglang_router:
        cmd.extend(["--disaggregation-mode", mode])

    # Add multi-node flags
    if num_nodes > 1:
        cmd.extend([
            "--dist-init-addr", f"{leader_ip}:{dist_init_port}",
            "--nnodes", str(num_nodes),
            "--node-rank", str(node_rank),
        ])

    # Add dump config
    if dump_config_path and not use_sglang_router:
        cmd.extend(["--dump-config-to", str(dump_config_path)])

    # Add config flags
    cmd.extend(config_to_cli_args(config))

    return cmd


def config_to_cli_args(config: dict[str, Any]) -> list[str]:
    """Convert config dict to CLI arguments.

    Args:
        config: Configuration dictionary

    Returns:
        List of CLI arguments
    """
    args: list[str] = []
    for key, value in sorted(config.items()):
        flag_name = key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(f"--{flag_name}")
        elif isinstance(value, list):
            args.append(f"--{flag_name}")
            args.extend(str(v) for v in value)
        elif value is not None:
            args.extend([f"--{flag_name}", str(value)])
    return args

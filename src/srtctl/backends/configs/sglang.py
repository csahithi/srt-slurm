# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SGLang backend configuration.

Implements BackendProtocol for SGLang inference serving with prefill/decode disaggregation.
"""

import builtins
from collections.abc import Sequence
from dataclasses import field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
)

from marshmallow import Schema
from marshmallow_dataclass import dataclass

if TYPE_CHECKING:
    from srtctl.core.endpoints import Endpoint, Process
    from srtctl.core.runtime import RuntimeContext

# Type alias for worker modes
WorkerMode = Literal["prefill", "decode", "agg"]


@dataclass(frozen=True)
class SGLangConfig:
    """SGLang worker configuration (prefill/decode/aggregated).

    Each mode can have its own configuration dict that gets converted
    to CLI flags when starting the worker.
    """

    prefill: dict[str, Any] | None = None
    decode: dict[str, Any] | None = None
    aggregated: dict[str, Any] | None = None

    Schema: ClassVar[type[Schema]] = Schema


@dataclass(frozen=True)
class SGLangBackendConfig:
    """SGLang backend configuration - implements BackendProtocol.

    This frozen dataclass both holds configuration AND implements the
    BackendProtocol methods for process allocation and launching.

    Example YAML:
        backend:
          type: sglang
          prefill_environment:
            CUDA_LAUNCH_BLOCKING: "1"
          sglang_config:
            prefill:
              mem-fraction-static: 0.8
              chunked-prefill-size: 8192
            decode:
              mem-fraction-static: 0.9
    """

    type: Literal["sglang"] = "sglang"
    gpu_type: str | None = None

    # Environment variables per mode
    prefill_environment: dict[str, str] = field(default_factory=dict)
    decode_environment: dict[str, str] = field(default_factory=dict)
    aggregated_environment: dict[str, str] = field(default_factory=dict)

    # SGLang-specific config
    sglang_config: SGLangConfig | None = None

    Schema: ClassVar[builtins.type[Schema]] = Schema

    # =========================================================================
    # BackendProtocol Implementation
    # =========================================================================

    def get_config_for_mode(self, mode: WorkerMode) -> dict[str, Any]:
        """Get merged config dict for a worker mode."""
        if not self.sglang_config:
            return {}

        if mode == "prefill":
            return dict(self.sglang_config.prefill or {})
        elif mode == "decode":
            return dict(self.sglang_config.decode or {})
        elif mode == "agg":
            return dict(self.sglang_config.aggregated or {})
        return {}

    def get_environment_for_mode(self, mode: WorkerMode) -> dict[str, str]:
        """Get environment variables for a worker mode."""
        if mode == "prefill":
            return dict(self.prefill_environment)
        elif mode == "decode":
            return dict(self.decode_environment)
        elif mode == "agg":
            return dict(self.aggregated_environment)
        return {}

    def allocate_endpoints(
        self,
        num_prefill: int,
        num_decode: int,
        num_agg: int,
        gpus_per_prefill: int,
        gpus_per_decode: int,
        gpus_per_agg: int,
        gpus_per_node: int,
        available_nodes: Sequence[str],
    ) -> list["Endpoint"]:
        """Allocate endpoints to nodes."""
        from srtctl.core.endpoints import allocate_endpoints

        return allocate_endpoints(
            num_prefill=num_prefill,
            num_decode=num_decode,
            num_agg=num_agg,
            gpus_per_prefill=gpus_per_prefill,
            gpus_per_decode=gpus_per_decode,
            gpus_per_agg=gpus_per_agg,
            gpus_per_node=gpus_per_node,
            available_nodes=available_nodes,
        )

    def endpoints_to_processes(
        self,
        endpoints: list["Endpoint"],
        base_port: int = 8081,
    ) -> list["Process"]:
        """Convert endpoints to processes."""
        from srtctl.core.endpoints import endpoints_to_processes

        return endpoints_to_processes(endpoints, base_port)

    def build_worker_command(
        self,
        process: "Process",
        endpoint_processes: list["Process"],
        runtime: "RuntimeContext",
        use_sglang_router: bool = False,
        dump_config_path: Path | None = None,
    ) -> list[str]:
        """Build the command to start an SGLang worker process."""
        from srtctl.core.runtime import get_hostname_ip

        mode = process.endpoint_mode
        config = self.get_config_for_mode(mode)

        # Determine if multi-node
        endpoint_nodes = list(dict.fromkeys(p.node for p in endpoint_processes))
        is_multi_node = len(endpoint_nodes) > 1

        # Get leader IP for distributed init
        leader_ip = get_hostname_ip(endpoint_nodes[0])
        dist_init_port = 29500

        # Choose Python module
        python_module = "sglang.launch_server" if use_sglang_router else "dynamo.sglang"

        # Get served model name from config
        served_model_name = runtime.model_path.name
        if self.sglang_config:
            for cfg in [self.sglang_config.prefill, self.sglang_config.aggregated]:
                if cfg:
                    name = cfg.get("served-model-name") or cfg.get("served_model_name")
                    if name:
                        served_model_name = name
                        break

        cmd = [
            "python3",
            "-m",
            python_module,
            "--model-path",
            str(runtime.model_path),
            "--served-model-name",
            served_model_name,
            "--host",
            "0.0.0.0",
        ]

        # Add disaggregation mode flag (not for agg mode)
        if mode != "agg" and not use_sglang_router:
            cmd.extend(["--disaggregation-mode", mode])

        # Add multi-node coordination flags
        if is_multi_node:
            node_rank = endpoint_nodes.index(process.node)
            cmd.extend(
                [
                    "--dist-init-addr",
                    f"{leader_ip}:{dist_init_port}",
                    "--nnodes",
                    str(len(endpoint_nodes)),
                    "--node-rank",
                    str(node_rank),
                ]
            )

        # Add config dump path
        if dump_config_path and not use_sglang_router:
            cmd.extend(["--dump-config-to", str(dump_config_path)])

        # Add all config flags
        cmd.extend(_config_to_cli_args(config))

        return cmd


def _config_to_cli_args(config: dict[str, Any]) -> list[str]:
    """Convert config dict to CLI arguments."""
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


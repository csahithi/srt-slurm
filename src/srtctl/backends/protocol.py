# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protocol definition for backend implementations."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from srtctl.core.endpoints import Endpoint, Process
    from srtctl.core.process_registry import NamedProcesses
    from srtctl.core.runtime import RuntimeContext


class BackendProtocol(Protocol):
    """Protocol that all backend configurations must implement.

    This allows frozen dataclasses to act as backends by implementing these methods.
    Each backend is responsible for:
    1. Allocating logical endpoints (serving units)
    2. Converting endpoints to physical processes
    3. Starting and managing those processes

    Backends currently supported:
    - SGLang: 1 process per node (ntasks = number of nodes in endpoint)

    Future backends:
    - TRT-LLM: 1 process per GPU (ntasks = total GPUs in endpoint)
    - vLLM: 1 process per GPU (ntasks = total GPUs in endpoint)
    """

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
        """Allocate logical endpoints based on backend-specific logic.

        Args:
            num_prefill: Number of prefill workers
            num_decode: Number of decode workers
            num_agg: Number of aggregated workers
            gpus_per_prefill: GPUs per prefill worker
            gpus_per_decode: GPUs per decode worker
            gpus_per_agg: GPUs per agg worker
            gpus_per_node: GPUs available per node
            available_nodes: Sequence of available node hostnames

        Returns:
            List of Endpoint objects with GPU allocations
        """
        ...

    def endpoints_to_processes(
        self,
        endpoints: list["Endpoint"],
        base_port: int = 8081,
    ) -> list["Process"]:
        """Convert logical endpoints to physical processes.

        Backend-specific mapping:
        - SGLang: 1 process per node (ntasks = number of nodes in endpoint)
        - TRT-LLM: 1 process per GPU (ntasks = total GPUs in endpoint)
        - vLLM: 1 process per GPU (ntasks = total GPUs in endpoint)

        Args:
            endpoints: List of logical endpoints
            base_port: Base port for DYN_SYSTEM_PORT assignment

        Returns:
            List of Process objects with individual process details
        """
        ...

    def start_processes(
        self,
        processes: list["Process"],
        runtime: "RuntimeContext",
        environment: dict[str, str],
    ) -> "NamedProcesses":
        """Start all processes for this backend.

        Args:
            processes: List of Process objects to start
            runtime: RuntimeContext with paths and node information
            environment: Environment variables from config

        Returns:
            Dictionary mapping process names to ManagedProcess objects
        """
        ...

    def get_config_for_mode(self, mode: str) -> dict[str, object]:
        """Get the merged config dict for a worker mode.

        Args:
            mode: "prefill", "decode", or "agg"

        Returns:
            Merged config dict (shared_config + mode-specific config)
        """
        ...

    def get_environment_for_mode(self, mode: str) -> dict[str, str]:
        """Get environment variables for a worker mode.

        Args:
            mode: "prefill", "decode", or "agg"

        Returns:
            Dictionary of environment variables
        """
        ...

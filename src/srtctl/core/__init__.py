# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Core modules for srtctl.

This package contains:
- config: Configuration loading and validation
- schema: Frozen dataclass schemas (SrtConfig, etc.)
- formatting: FormattablePath and FormattableString for deferred expansion
- runtime: RuntimeContext for computed paths and values
- endpoints: Endpoint and Process dataclasses
- process_registry: Process lifecycle management
- utils: Helper functions (srun, wait_for_port, etc.)
"""

# Re-export backend configs from their new location
from srtctl.backends.configs import (
    BackendConfig,
    BackendProtocol,
    BackendType,
    SGLangBackendConfig,
    SGLangConfig,
)

from .config import get_srtslurm_setting, load_config
from .endpoints import Endpoint, Process, allocate_endpoints, endpoints_to_processes
from .formatting import FormattablePath, FormattableString
from .process_registry import (
    ManagedProcess,
    NamedProcesses,
    ProcessRegistry,
    setup_signal_handlers,
    start_process_monitor,
)
from .runtime import Nodes, RuntimeContext, get_hostname_ip, get_slurm_job_id
from .schema import (
    BenchmarkConfig,
    ClusterConfig,
    FrontendConfig,
    HealthCheckConfig,
    ModelConfig,
    OutputConfig,
    ProfilingConfig,
    ResourceConfig,
    SlurmConfig,
    SrtConfig,
)

__all__ = [
    # Config loading
    "load_config",
    "get_srtslurm_setting",
    # Schema types (frozen dataclasses)
    "SrtConfig",
    "ResourceConfig",
    "BenchmarkConfig",
    "FrontendConfig",
    "ProfilingConfig",
    "ModelConfig",
    "SlurmConfig",
    "OutputConfig",
    "HealthCheckConfig",
    "ClusterConfig",
    # Backend configs (re-exported from backends.configs)
    "SGLangBackendConfig",
    "SGLangConfig",
    "BackendConfig",
    "BackendProtocol",
    "BackendType",
    # Formatting
    "FormattablePath",
    "FormattableString",
    # Runtime
    "Nodes",
    "RuntimeContext",
    "get_slurm_job_id",
    "get_hostname_ip",
    # Endpoints
    "Endpoint",
    "Process",
    "allocate_endpoints",
    "endpoints_to_processes",
    # Process management
    "ManagedProcess",
    "NamedProcesses",
    "ProcessRegistry",
    "setup_signal_handlers",
    "start_process_monitor",
]

"""
srtctl - Benchmark submission framework for distributed serving workloads.

This package provides Python-first orchestration for LLM inference benchmarks
on SLURM clusters, supporting multiple backends:
- SGLang (with prefill/decode disaggregation)
- vLLM (placeholder)
- TensorRT-LLM (placeholder)

Key modules:
- core.config: Configuration loading and validation
- core.schema: Frozen dataclass definitions (SrtConfig, etc.)
- core.runtime: RuntimeContext for computed paths and values
- core.endpoints: Endpoint and Process dataclasses for worker topology
- core.process_registry: Process lifecycle management
- backends.configs: Backend-specific configuration dataclasses
- cli.submit: Job submission interface
- cli.do_sweep: Main orchestration script
- logging_utils: Consistent logging and emoji helpers

Usage:
    # Submit with orchestrator (Python-controlled)
    srtctl apply -f config.yaml
"""

__version__ = "0.3.0"

# Logging utilities (should be first)
# Backend configs
from .backends.configs import (
    BackendConfig,
    BackendProtocol,
    BackendType,
    SGLangBackendConfig,
)

# Core modules
from .core.config import get_srtslurm_setting, load_config
from .core.endpoints import Endpoint, Process, allocate_endpoints, endpoints_to_processes
from .core.formatting import FormattablePath, FormattableString
from .core.process_registry import (
    ManagedProcess,
    NamedProcesses,
    ProcessRegistry,
)
from .core.runtime import Nodes, RuntimeContext, get_hostname_ip, get_slurm_job_id
from .core.schema import SrtConfig
from .logging_utils import (
    CHECK,
    CROSS,
    GEAR,
    HOURGLASS,
    PACKAGE,
    ROCKET,
    WARN,
    WRENCH,
    error,
    get_logger,
    section,
    setup_logging,
    step,
    success,
    waiting,
    warn,
)

__all__ = [
    # Version
    "__version__",
    # Logging
    "CHECK",
    "CROSS",
    "GEAR",
    "HOURGLASS",
    "PACKAGE",
    "ROCKET",
    "WARN",
    "WRENCH",
    "get_logger",
    "section",
    "setup_logging",
    "step",
    "success",
    "error",
    "warn",
    "waiting",
    # Config
    "load_config",
    "get_srtslurm_setting",
    "SrtConfig",
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
    # Backends
    "BackendProtocol",
    "BackendConfig",
    "BackendType",
    "SGLangBackendConfig",
]

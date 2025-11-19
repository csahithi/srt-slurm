"""
InfBench - Benchmark submission framework for distributed serving workloads.
"""

__version__ = "0.1.0"

from .core.config import load_config
from .backends.sglang import generate_sglang_config_file

__all__ = [
    "load_config",
    "generate_sglang_config_file",
]

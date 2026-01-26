# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Parser protocols and registries for benchmark and node log parsing.
This module provides extensible parsing infrastructure:
- BenchmarkParser: Parses benchmark.out files based on benchmark type
- NodeParser: Parses prefill/decode/agg logs based on backend type
Usage:
    from analysis.srtlog.parsers import get_benchmark_parser, get_node_parser
    # Get parser by type
    bench_parser = get_benchmark_parser("sa-bench")
    results = bench_parser.parse(benchmark_out_path)
    node_parser = get_node_parser("sglang")
    nodes = node_parser.parse_logs(log_dir)
"""

from pathlib import Path
from typing import Any, Protocol

from analysis.srtlog.models import BenchmarkLaunchCommand, NodeLaunchCommand, NodeMetrics


class BenchmarkParserProtocol(Protocol):
    """Protocol for benchmark output parsers.
    Each benchmark type (sa-bench, mooncake-router, etc.) should have
    a parser that implements this protocol.
    """

    @property
    def benchmark_type(self) -> str:
        """Return the benchmark type this parser handles."""
        ...

    def parse(self, benchmark_out_path: Path) -> dict[str, Any]:
        """Parse benchmark.out file and return results.
        Args:
            benchmark_out_path: Path to the benchmark.out file
        Returns:
            Dict with benchmark results including:
            - output_tps: Output tokens per second
            - mean_ttft_ms: Mean time to first token
            - mean_itl_ms: Mean inter-token latency
            - etc.
        """
        ...

    def parse_launch_command(self, log_content: str) -> BenchmarkLaunchCommand | None:
        """Parse the benchmark launch command from log content.
        Args:
            log_content: Content of the benchmark log file
        Returns:
            BenchmarkLaunchCommand with parsed parameters, or None if not found
        """
        ...

    def parse_result_json(self, json_path: Path) -> dict[str, Any]:
        """Parse a benchmark result JSON file.
        Args:
            json_path: Path to a result JSON file
        Returns:
            Dict with parsed benchmark metrics
        """
        ...

    def find_result_directory(self, run_path: Path, isl: int | None = None, osl: int | None = None) -> Path | None:
        """Find the directory containing benchmark results within a run directory.
        
        This method encapsulates the logic for locating result files, which varies by benchmark type.
        For example:
        - sa-bench: looks for directories like "sa-bench_isl_8192_osl_1024"
        - mooncake-router: looks in "logs/artifacts/" subdirectory
        
        Args:
            run_path: Path to the run directory (contains logs/, metadata, etc.)
            isl: Input sequence length (optional, used for pattern matching)
            osl: Output sequence length (optional, used for pattern matching)
        
        Returns:
            Path to directory containing result files, or None if not found
        """
        ...
    
    def parse_result_directory(self, result_dir: Path) -> list[dict[str, Any]]:
        """Parse all result files in a directory.
        
        Args:
            result_dir: Directory containing benchmark result files
            
        Returns:
            List of result dicts (one per concurrency level or benchmark run)
        """
        ...


class NodeParserProtocol(Protocol):
    """Protocol for node log parsers.
    Each backend type (sglang, trtllm, etc.) should have a parser
    that implements this protocol for parsing prefill/decode/agg logs.
    """

    @property
    def backend_type(self) -> str:
        """Return the backend type this parser handles."""
        ...

    def parse_logs(self, log_dir: Path) -> list[NodeMetrics]:
        """Parse all node logs in a directory.
        Args:
            log_dir: Directory containing prefill/decode/agg .out/.err files
        Returns:
            List of NodeMetrics objects, one per worker
        """
        ...

    def parse_single_log(self, log_path: Path) -> NodeMetrics | None:
        """Parse a single node log file.
        Args:
            log_path: Path to a prefill/decode/agg log file
        Returns:
            NodeMetrics object or None if parsing failed
        """
        ...

    def parse_launch_command(self, log_content: str, worker_type: str = "unknown") -> NodeLaunchCommand | None:
        """Parse the worker launch command from log content.
        Args:
            log_content: Content of the worker log file
            worker_type: Type of worker (prefill, decode, agg)
        Returns:
            NodeLaunchCommand with parsed parameters, or None if not found
        """
        ...


# Registry for benchmark parsers
_benchmark_parsers: dict[str, type] = {}

# Registry for node parsers
_node_parsers: dict[str, type] = {}


def register_benchmark_parser(benchmark_type: str):
    """Decorator to register a benchmark parser.
    Usage:
        @register_benchmark_parser("sa-bench")
        class SABenchParser:
            ...
    """

    def decorator(cls):
        _benchmark_parsers[benchmark_type] = cls
        return cls

    return decorator


def register_node_parser(backend_type: str):
    """Decorator to register a node parser.
    Usage:
        @register_node_parser("sglang")
        class SGLangNodeParser:
            ...
    """

    def decorator(cls):
        _node_parsers[backend_type] = cls
        return cls

    return decorator


def get_benchmark_parser(benchmark_type: str) -> BenchmarkParserProtocol:
    """Get a benchmark parser by type.
    Args:
        benchmark_type: Type of benchmark (e.g., "sa-bench", "mooncake-router")
    Returns:
        Instance of the appropriate benchmark parser
    Raises:
        ValueError: If no parser registered for the benchmark type
    """
    if benchmark_type not in _benchmark_parsers:
        available = ", ".join(_benchmark_parsers.keys()) or "none"
        raise ValueError(f"No benchmark parser registered for '{benchmark_type}'. Available: {available}")
    return _benchmark_parsers[benchmark_type]()


def get_node_parser(backend_type: str) -> NodeParserProtocol:
    """Get a node parser by backend type.
    Args:
        backend_type: Type of backend (e.g., "sglang", "trtllm")
    Returns:
        Instance of the appropriate node parser
    Raises:
        ValueError: If no parser registered for the backend type
    """
    if backend_type not in _node_parsers:
        available = ", ".join(_node_parsers.keys()) or "none"
        raise ValueError(f"No node parser registered for '{backend_type}'. Available: {available}")
    return _node_parsers[backend_type]()


def list_benchmark_parsers() -> list[str]:
    """List all registered benchmark parser types."""
    return list(_benchmark_parsers.keys())


def list_node_parsers() -> list[str]:
    """List all registered node parser types."""
    return list(_node_parsers.keys())


# Import parsers to trigger registration
from analysis.srtlog.parsers.benchmark import *  # noqa: E402, F401, F403
from analysis.srtlog.parsers.nodes import *  # noqa: E402, F401, F403

# Re-export models for convenience
__all__ = [
    "BenchmarkLaunchCommand",
    "NodeLaunchCommand",
    "NodeMetrics",
    "BenchmarkParserProtocol",
    "NodeParserProtocol",
    "get_benchmark_parser",
    "get_node_parser",
    "list_benchmark_parsers",
    "list_node_parsers",
    "register_benchmark_parser",
    "register_node_parser",
]
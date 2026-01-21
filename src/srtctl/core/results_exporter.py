# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Results exporter for benchmark runs.

Rolls up benchmark datapoints into CSV and Parquet formats after experiment completion.
Parquet is preferred for web transmission due to better compression.

Uses RunLoader and NodeAnalyzer from the analysis package for comprehensive log parsing.
"""

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
import traceback
from typing import TYPE_CHECKING, Any, Optional
import json
import pandas as pd
from pydantic import BaseModel, Field, model_validator
import yaml
import os
import numpy as np
import sys

from analysis.srtlog import BenchmarkRun, ProfilerMetadata, RunLoader
from analysis.srtlog.cache_manager import CacheManager
from analysis.srtlog.log_parser import NodeAnalyzer
from analysis.srtlog.models import NodeMetrics, ProfilerResults, RunMetadata
from srtctl.core.runtime import RuntimeContext
from srtctl.core.schema import SrtConfig

logger = logging.getLogger(__name__)


class ReportDatapoint(BaseModel):
    """Single datapoint from a benchmark run."""

    # Run metadata (from RunMetadata)
    job_id: str
    job_name: str  # run name
    run_date: str
    profiler_type: str  # benchmark type

    # Resource config (from RunMetadata)
    gpu_type: str
    gpus_per_node: int
    prefill_nodes: int
    decode_nodes: int
    prefill_workers: int
    decode_workers: int
    agg_workers: int
    
    # Computed - must be provided explicitly
    total_gpus: int

    # Benchmark params (from ProfilerMetadata)
    isl: int
    osl: int
    concurrency: int  # from ProfilerResults.get_datapoint()
    concurrencies: list[int] | str  # expected concurrencies from ProfilerMetadata
    request_rate: float | str | None

    # Throughput metrics
    output_tps: float
    total_tps: float | None = None
    request_throughput: float | None = None
    request_goodput: float | None = None

    # Latency metrics - mean
    mean_ttft_ms: float | None = None
    mean_tpot_ms: float | None = None
    mean_itl_ms: float | None = None
    mean_e2el_ms: float | None = None

    # Latency metrics - median
    median_ttft_ms: float | None = None
    median_tpot_ms: float | None = None
    median_itl_ms: float | None = None
    median_e2el_ms: float | None = None

    # Latency metrics - p99
    p99_ttft_ms: float | None = None
    p99_tpot_ms: float | None = None
    p99_itl_ms: float | None = None
    p99_e2el_ms: float | None = None

    # Token counts
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None

    # Run metadata
    duration: float | None = None
    completed: int | None = None
    num_prompts: int | None = None

    # Computed metrics
    output_tps_per_gpu: float | None = None
    total_tps_per_gpu: float | None = None
    output_tps_per_user: float | None = None

    # Node metrics (shared across all datapoints from the same run)
    node_metrics: list[NodeMetrics] = Field(default_factory=list)

    @model_validator(mode="after")
    def compute_derived_metrics(self) -> "ReportDatapoint":
        """Compute derived metrics."""
        if self.total_gpus > 0:
            self.output_tps_per_gpu = self.output_tps / self.total_gpus
            if self.total_tps is not None:
                self.total_tps_per_gpu = self.total_tps / self.total_gpus

        if self.mean_tpot_ms is not None and self.mean_tpot_ms > 0:
            self.output_tps_per_user = 1000 / self.mean_tpot_ms
        
        return self

    def summarize_node_metrics(self) -> dict[str, Any]:
        """Summarize node metrics for DataFrame inclusion. 

        Args:
            node_metrics: List of NodeMetrics objects

        Returns:
            Dict with summary statistics
        """
        summary: dict[str, Any] = {
            "num_nodes": len(self.node_metrics),
            "num_prefill_nodes": sum(1 for n in self.node_metrics if n.is_prefill),
            "num_decode_nodes": sum(1 for n in self.node_metrics if n.is_decode),
            "total_batches": sum(len(n.batches) for n in self.node_metrics),
            "total_memory_snapshots": sum(len(n.memory_snapshots) for n in self.node_metrics),
        }

        # Aggregate batch metrics across all nodes
        all_batches = [b for n in self.node_metrics for b in n.batches]
        if all_batches:
            # Filter for batches that have batch_size attribute
            batch_sizes = [b.batch_size for b in all_batches if hasattr(b, "batch_size") and b.batch_size is not None]
            running_reqs = [b.running_req for b in all_batches if b.running_req is not None]
            queue_reqs = [b.queue_req for b in all_batches if b.queue_req is not None]
            
            if batch_sizes:
                summary["avg_batch_size"] = sum(batch_sizes) / len(batch_sizes)
            if running_reqs:
                summary["avg_running_reqs"] = sum(running_reqs) / len(running_reqs)
            if queue_reqs:
                summary["avg_pending_reqs"] = sum(queue_reqs) / len(queue_reqs)

        # Aggregate memory metrics across all nodes
        all_memory = [m for n in self.node_metrics for m in n.memory_snapshots]
        if all_memory:
            # Memory usage (filter out None values)
            mem_values = [m.mem_usage_gb for m in all_memory if m.mem_usage_gb is not None]
            if mem_values:
                summary["avg_mem_usage_gb"] = sum(mem_values) / len(mem_values)
                summary["peak_mem_usage_gb"] = max(mem_values)

            # KV cache usage (filter out None values)
            kv_values = [m.kv_cache_gb for m in all_memory if m.kv_cache_gb is not None]
            if kv_values:
                summary["avg_kv_cache_gb"] = sum(kv_values) / len(kv_values)
                summary["peak_kv_cache_gb"] = max(kv_values)

        # Extract prefill config (from first prefill node - same for all)
        prefill_nodes = [n for n in self.node_metrics if n.is_prefill]
        if prefill_nodes and prefill_nodes[0].config:
            config = prefill_nodes[0].config
            # Handle both NodeConfig dataclass and legacy dict
            if hasattr(config, "server_args"):
                # NodeConfig dataclass - server_args contains flattened config
                server_args = config.server_args or {}
                summary.update({
                    f"prefill_config_{k}": v for k, v in server_args.items()
                    if k != "command"
                })
                # Include environment variables
                if hasattr(config, "environment") and config.environment:
                    summary.update({
                        f"prefill_env_{k}": v for k, v in config.environment.items()
                    })
            elif isinstance(config, dict) and "trtllm_config" in config:
                # Legacy dict format
                summary.update({
                    f"prefill_config_{k}": v for k, v in config["trtllm_config"].items()
                    if k != "command"
                })

            # Concatenate all commands from all prefill nodes
            prefill_commands = []
            for n in prefill_nodes:
                if n.config and hasattr(n.config, "server_args") and n.config.server_args:
                    cmd = n.config.server_args.get("command")
                    if cmd:
                        prefill_commands.append(f"{n.node_name}: {cmd}")
            if prefill_commands:
                summary["prefill_command"] = "\n".join(prefill_commands)

        # Extract decode config (from first decode node - same for all)
        decode_nodes = [n for n in self.node_metrics if n.is_decode]
        if decode_nodes and decode_nodes[0].config:
            config = decode_nodes[0].config
            # Handle both NodeConfig dataclass and legacy dict
            if hasattr(config, "server_args"):
                # NodeConfig dataclass - server_args contains flattened config
                server_args = config.server_args or {}
                summary.update({
                    f"decode_config_{k}": v for k, v in server_args.items()
                    if k != "command"
                })
                # Include environment variables
                if hasattr(config, "environment") and config.environment:
                    summary.update({
                        f"decode_env_{k}": v for k, v in config.environment.items()
                    })
            elif isinstance(config, dict) and "trtllm_config" in config:
                # Legacy dict format
                summary.update({
                    f"decode_config_{k}": v for k, v in config["trtllm_config"].items()
                    if k != "command"
                })

            # Concatenate all commands from all decode nodes
            decode_commands = []
            for n in decode_nodes:
                if n.config and hasattr(n.config, "server_args") and n.config.server_args:
                    cmd = n.config.server_args.get("command")
                    if cmd:
                        decode_commands.append(f"{n.node_name}: {cmd}")
            if decode_commands:
                summary["decode_command"] = "\n".join(decode_commands)

        return summary

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the report datapoint."""
        ret = self.model_dump()
        ret.pop("node_metrics")
        ret.update(self.summarize_node_metrics())   # this is a dict
        return ret


@dataclass
class ResultsExporter:
    """Exports benchmark results to CSV and Parquet formats.

    Usage:
        # With config and runtime (from srtctl orchestrator)
        exporter = ResultsExporter(config=config, runtime=runtime)
        exporter.export()

        # Standalone mode (just log directory)
        exporter = ResultsExporter.from_log_dir("/path/to/logs")
        df = exporter.to_dataframe()
    """

    config: Optional[SrtConfig] = None
    runtime: Optional[RuntimeContext] = None
    log_dir: Optional[str] = None
    datapoints: list[ReportDatapoint] = field(default_factory=list)

    @classmethod
    def from_log_dir(cls, log_dir: str) -> "ResultsExporter":
        """Create a standalone exporter from a log directory.

        Args:
            log_dir: Path to the logs directory

        Returns:
            ResultsExporter instance
        """
        return cls(log_dir=log_dir)

    def _get_log_dir(self) -> str:
        """Get the log directory path."""
        if self.log_dir:
            return self.log_dir

        if self.runtime:
            return str(self.runtime.log_dir)
        raise ValueError("Either log_dir or runtime must be provided")

    def collect_results(self) -> list[ReportDatapoint]:
        """Collect benchmark results from the log directory.

        Uses RunLoader to parse benchmark runs from the log directory.

        Returns:
            List of ReportDatapoint objects
        """
        log_dir = self._get_log_dir()

        # Use RunLoader to find and load benchmark runs
        loader = RunLoader(log_dir)
        runs, skipped = loader.load_all_with_skipped()

        if not runs:
            logger.warning(f"No benchmark runs found in {log_dir}")
            if skipped:
                logger.info(f"Skipped {len(skipped)} runs:")
                for skipped_run in skipped[:5]:
                    logger.info(f"  {skipped_run.run_id_path_pair.job_id} ({skipped_run.run_id_path_pair.run_dir}): {skipped_run.reason}")
            return []

        datapoints = []
        for run in runs:
            points = self._convert_run_to_datapoints(run)
            datapoints.extend(points)

        self.datapoints = datapoints
        logger.info(f"Collected {len(datapoints)} datapoints from {len(runs)} benchmark runs")
        return datapoints

    def _convert_run_to_datapoints(self, run: BenchmarkRun) -> list[ReportDatapoint]:
        """Convert a BenchmarkRun to ReportDatapoint objects.

        Each concurrency level in the run becomes a separate datapoint.
        Node metrics are fetched once and shared across all datapoints from the same run.

        Args:
            run: BenchmarkRun object from RunLoader

        Returns:
            List of ReportDatapoint objects
        """
        datapoints = []
        run_metadata: RunMetadata = run.metadata
        profiler_results: ProfilerResults = run.profiler_results
        profiler_metadata: ProfilerMetadata = run.profiler_metadata

        # Fetch node metrics for this run (shared across all datapoints)
        node_metrics: list[NodeMetrics] = self._get_node_metrics_for_run(run_metadata.path)

        # Create a datapoint for each concurrency level
        num_results = len(profiler_results.concurrency_values)
        for i in range(num_results):
            datapoint = ReportDatapoint(
                **run_metadata.model_dump(),
                **profiler_metadata.model_dump(),
                **profiler_results.get_datapoint(i),
                # Computed fields not in model_dump()
                total_gpus=run_metadata.total_gpus,
                # Node metrics (shared across all concurrency levels in this run)
                node_metrics=node_metrics,
            )
            datapoints.append(datapoint)

        # Sort by concurrency
        datapoints.sort(key=lambda x: x.concurrency)
        return datapoints

    def _get_node_metrics_for_run(self, run_path: Path) -> list[NodeMetrics]:
        """Get node metrics for a specific run directory.

        Args:
            run_path: Path to the run directory

        Returns:
            List of NodeMetrics objects
        """
        try:
            analyzer = NodeAnalyzer()
            metrics = analyzer.parse_run_logs(run_path, return_dicts=False)
            return metrics
        except Exception as e:
            logger.warning(f"Failed to parse node metrics for {run_path}: {traceback.format_exc()}")
            return []

    def to_dataframe(self, include_node_metrics: bool = True):
        """Convert datapoints to pandas DataFrame.

        Args:
            include_node_metrics: If True, include node metrics summary columns

        Returns:
            pandas DataFrame with all datapoints
        """
        if not self.datapoints:
            self.collect_results()

        if not self.datapoints:
            return pd.DataFrame()

        # Convert datapoints to summaries
        rows = [dp.get_summary() for dp in self.datapoints]
        return pd.DataFrame(rows)
    
    def export(self, output_dir: Path | None = None) -> dict[str, Path]:
        """Export results to CSV and Parquet formats.

        Parquet is better for web transmission due to columnar compression.

        Args:
            output_dir: Output directory (defaults to runtime.log_dir)

        Returns:
            Dict with paths to exported files {"csv": path, "parquet": path}
        """
        if output_dir is None:
            output_dir = Path(self._get_log_dir())
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe()

        if df.empty:
            logger.warning("No datapoints to export")
            return {}

        # Generate filenames
        if self.runtime:
            csv_path = output_dir / f"{self.runtime.job_id}_results.csv"
            parquet_path = output_dir / f"{self.runtime.job_id}_results.parquet"
        else:
            csv_path = output_dir / "results.csv"
            parquet_path = output_dir / "results.parquet"

        exported = {}

        # Export CSV
        try:
            df.to_csv(csv_path, index=False)
            logger.info(f"Exported CSV: {csv_path}")
            exported["csv"] = csv_path
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")

        # Export Parquet (better for web transmission)
        try:
            df.to_parquet(parquet_path, index=False, compression="snappy")
            logger.info(f"Exported Parquet: {parquet_path}")
            exported["parquet"] = parquet_path
        except Exception as e:
            logger.error(f"Failed to export Parquet: {e}")

        return exported

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the benchmark results.

        Returns:
            Dict with summary statistics
        """
        if not self.datapoints:
            return {}

        # Find peak throughput datapoint
        peak_dp = max(self.datapoints, key=lambda x: x.output_tps)

        return {
            "job_id": self.runtime.job_id if self.runtime else peak_dp.job_id,
            "run_name": self.runtime.run_name if self.runtime else peak_dp.run_name,
            "benchmark_type": self.config.benchmark.type if self.config else peak_dp.benchmark_type,
            "num_datapoints": len(self.datapoints),
            "concurrencies": [dp.concurrency for dp in self.datapoints],
            "peak_output_tps": peak_dp.output_tps,
            "peak_concurrency": peak_dp.concurrency,
            "peak_output_tps_per_gpu": peak_dp.output_tps_per_gpu,
            "total_gpus": peak_dp.total_gpus,
        }


def export_results(config: "SrtConfig", runtime: "RuntimeContext") -> dict[str, Path]:
    """Convenience function to export benchmark results.

    Args:
        config: Job configuration
        runtime: Runtime context

    Returns:
        Dict with paths to exported files
    """
    exporter = ResultsExporter(config=config, runtime=runtime)
    return exporter.export()


def clear_cache(log_dir: str) -> None:
    """Clear all analysis caches for a log directory.

    Clears both the top-level cache and individual run caches.

    Args:
        log_dir: Path to the logs directory
    """

    # Clear top-level cache
    cache_mgr = CacheManager(log_dir)
    cache_mgr.invalidate_cache()
    logger.info(f"Cleared cache for {log_dir}")

    # Clear individual run caches
    cleared_count = 0
    for entry in os.listdir(log_dir):
        run_path = os.path.join(log_dir, entry)
        if os.path.isdir(run_path):
            try:
                run_cache = CacheManager(run_path)
                run_cache.invalidate_cache()
                cleared_count += 1
            except Exception:
                pass

    logger.info(f"Cleared {cleared_count} run caches")


def create_args_parser() -> argparse.ArgumentParser:
    """Create an argument parser for the results exporter."""
    parser = argparse.ArgumentParser(description="Results exporter utilities")
    parser.add_argument(
        "--log-dir",
        type=str,
        required=True,
        help="Path to the logs directory",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear all analysis caches",
    )
    return parser

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = create_args_parser()

    args = parser.parse_args()

    if args.clear_cache:
        clear_cache(args.log_dir)
        logger.info(f"Cache cleared for {args.log_dir}")

    # Use ResultsExporter in standalone mode
    exporter = ResultsExporter.from_log_dir(args.log_dir)
    exporter.collect_results()

    if not exporter.datapoints:
        logger.error("No datapoints found")
        sys.exit(0)

    logger.info(f"Found {len(exporter.datapoints)} datapoints")

    # Export to CSV and Parquet
    exported = exporter.export()

    for fmt, path in exported.items():
        logger.info(f"Exported {fmt.upper()}: {path}")
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
from typing import TYPE_CHECKING, Any
import json
import pandas as pd
import yaml
import os
import numpy as np
import sys

from analysis.srtlog import BenchmarkRun, RunLoader
from analysis.srtlog.cache_manager import CacheManager
from analysis.srtlog.log_parser import NodeAnalyzer
from analysis.srtlog.models import NodeMetrics
if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkDatapoint:
    """Single datapoint from a benchmark run."""

    # Run metadata
    job_id: str
    run_name: str
    run_date: str
    benchmark_type: str

    # Resource config
    gpu_type: str
    gpus_per_node: int
    prefill_nodes: int
    decode_nodes: int
    prefill_workers: int
    decode_workers: int
    agg_workers: int
    total_gpus: int

    # Benchmark params
    isl: int
    osl: int
    concurrency: int
    request_rate: float | None

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
    node_metrics: list[NodeMetrics] = field(default_factory=list)

    def __post_init__(self):
        """Compute derived metrics."""
        if self.total_gpus > 0:
            self.output_tps_per_gpu = self.output_tps / self.total_gpus
            if self.total_tps is not None:
                self.total_tps_per_gpu = self.total_tps / self.total_gpus

        if self.mean_tpot_ms is not None and self.mean_tpot_ms > 0:
            self.output_tps_per_user = 1000 / self.mean_tpot_ms


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

    config: "SrtConfig | None" = None
    runtime: "RuntimeContext | None" = None
    log_dir: str | None = None
    datapoints: list[BenchmarkDatapoint] = field(default_factory=list)

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

    def collect_results(self) -> list[BenchmarkDatapoint]:
        """Collect benchmark results from the log directory.

        Uses RunLoader to parse benchmark runs from the log directory.

        Returns:
            List of BenchmarkDatapoint objects
        """
        log_dir = self._get_log_dir()

        # Use RunLoader to find and load benchmark runs
        loader = RunLoader(log_dir)
        runs, skipped = loader.load_all_with_skipped()

        if not runs:
            logger.warning(f"No benchmark runs found in {log_dir}")
            if skipped:
                logger.info(f"Skipped {len(skipped)} runs:")
                for job_id, run_dir, reason in skipped[:5]:
                    logger.info(f"  {job_id} ({run_dir}): {reason}")
            return []

        datapoints = []
        for run in runs:
            points = self._convert_run_to_datapoints(run)
            datapoints.extend(points)

        self.datapoints = datapoints
        logger.info(f"Collected {len(datapoints)} datapoints from {len(runs)} benchmark runs")
        return datapoints

    def _convert_run_to_datapoints(self, run: BenchmarkRun) -> list[BenchmarkDatapoint]:
        """Convert a BenchmarkRun to BenchmarkDatapoint objects.

        Each concurrency level in the run becomes a separate datapoint.
        Node metrics are fetched once and shared across all datapoints from the same run.

        Args:
            run: BenchmarkRun object from RunLoader

        Returns:
            List of BenchmarkDatapoint objects
        """
        datapoints = []
        meta = run.metadata
        profiler = run.profiler

        # Calculate total GPUs
        if meta.mode == "disaggregated":
            total_gpus = (meta.prefill_nodes * meta.gpus_per_node) + (meta.decode_nodes * meta.gpus_per_node)
        else:
            total_gpus = meta.agg_nodes * meta.gpus_per_node

        # Fetch node metrics for this run (shared across all datapoints)
        node_metrics = self._get_node_metrics_for_run(meta.path)

        # Create a datapoint for each concurrency level
        num_results = len(profiler.output_tps)
        for i in range(num_results):
            datapoint = BenchmarkDatapoint(
                # Run metadata
                job_id=meta.job_id,
                run_name=meta.job_name or (self.runtime.run_name if self.runtime else ""),
                run_date=meta.run_date,
                benchmark_type=profiler.profiler_type,
                # Resource config
                gpu_type=meta.gpu_type,
                gpus_per_node=meta.gpus_per_node,
                prefill_nodes=meta.prefill_nodes,
                decode_nodes=meta.decode_nodes,
                prefill_workers=meta.prefill_workers,
                decode_workers=meta.decode_workers,
                agg_workers=meta.agg_workers,
                total_gpus=total_gpus,
                # Benchmark params
                isl=int(profiler.isl) if profiler.isl else 0,
                osl=int(profiler.osl) if profiler.osl else 0,
                concurrency=profiler.concurrency_values[i] if i < len(profiler.concurrency_values) else 0,
                request_rate=profiler.request_rate[i] if i < len(profiler.request_rate) else None,
                # Throughput metrics
                output_tps=profiler.output_tps[i],
                total_tps=profiler.total_tps[i] if i < len(profiler.total_tps) else None,
                request_throughput=profiler.request_throughput[i] if i < len(profiler.request_throughput) else None,
                request_goodput=profiler.request_goodput[i] if i < len(profiler.request_goodput) else None,
                # Mean latencies
                mean_ttft_ms=profiler.mean_ttft_ms[i] if i < len(profiler.mean_ttft_ms) else None,
                mean_tpot_ms=profiler.mean_tpot_ms[i] if i < len(profiler.mean_tpot_ms) else None,
                mean_itl_ms=profiler.mean_itl_ms[i] if i < len(profiler.mean_itl_ms) else None,
                mean_e2el_ms=profiler.mean_e2el_ms[i] if i < len(profiler.mean_e2el_ms) else None,
                # Median latencies
                median_ttft_ms=profiler.median_ttft_ms[i] if i < len(profiler.median_ttft_ms) else None,
                median_tpot_ms=profiler.median_tpot_ms[i] if i < len(profiler.median_tpot_ms) else None,
                median_itl_ms=profiler.median_itl_ms[i] if i < len(profiler.median_itl_ms) else None,
                median_e2el_ms=profiler.median_e2el_ms[i] if i < len(profiler.median_e2el_ms) else None,
                # P99 latencies
                p99_ttft_ms=profiler.p99_ttft_ms[i] if i < len(profiler.p99_ttft_ms) else None,
                p99_tpot_ms=profiler.p99_tpot_ms[i] if i < len(profiler.p99_tpot_ms) else None,
                p99_itl_ms=profiler.p99_itl_ms[i] if i < len(profiler.p99_itl_ms) else None,
                p99_e2el_ms=profiler.p99_e2el_ms[i] if i < len(profiler.p99_e2el_ms) else None,
                # Token counts
                total_input_tokens=profiler.total_input_tokens[i] if i < len(profiler.total_input_tokens) else None,
                total_output_tokens=profiler.total_output_tokens[i] if i < len(profiler.total_output_tokens) else None,
                # Run metadata
                duration=profiler.duration[i] if i < len(profiler.duration) else None,
                completed=profiler.completed[i] if i < len(profiler.completed) else None,
                num_prompts=profiler.num_prompts[i] if i < len(profiler.num_prompts) else None,
                # Node metrics (shared across all concurrency levels in this run)
                node_metrics=node_metrics,
            )
            datapoints.append(datapoint)

        # Sort by concurrency
        datapoints.sort(key=lambda x: x.concurrency)
        return datapoints

    def _get_node_metrics_for_run(self, run_path: str) -> list[NodeMetrics]:
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

    def collect_node_metrics(self) -> list[dict[str, Any]]:
        """Collect node-level metrics from log files.

        Uses NodeAnalyzer to parse .err/.out files for detailed metrics.

        Returns:
            List of node metrics dicts
        """
        log_dir = self._get_log_dir()
        analyzer = NodeAnalyzer()
        return analyzer.parse_run_logs(log_dir, return_dicts=True)

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

        # Convert datapoints to dicts
        rows = []
        for dp in self.datapoints:
            row = {
                # Run metadata
                "job_id": dp.job_id,
                "run_name": dp.run_name,
                "run_date": dp.run_date,
                "benchmark_type": dp.benchmark_type,
                # Resource config
                "gpu_type": dp.gpu_type,
                "gpus_per_node": dp.gpus_per_node,
                "prefill_nodes": dp.prefill_nodes,
                "decode_nodes": dp.decode_nodes,
                "prefill_workers": dp.prefill_workers,
                "decode_workers": dp.decode_workers,
                "agg_workers": dp.agg_workers,
                "total_gpus": dp.total_gpus,
                # Benchmark params
                "isl": dp.isl,
                "osl": dp.osl,
                "concurrency": dp.concurrency,
                "request_rate": dp.request_rate,
                # Throughput metrics
                "output_tps": dp.output_tps,
                "total_tps": dp.total_tps,
                "request_throughput": dp.request_throughput,
                "request_goodput": dp.request_goodput,
                # Computed throughput metrics
                "output_tps_per_gpu": dp.output_tps_per_gpu,
                "total_tps_per_gpu": dp.total_tps_per_gpu,
                "output_tps_per_user": dp.output_tps_per_user,
                # Mean latencies
                "mean_ttft_ms": dp.mean_ttft_ms,
                "mean_tpot_ms": dp.mean_tpot_ms,
                "mean_itl_ms": dp.mean_itl_ms,
                "mean_e2el_ms": dp.mean_e2el_ms,
                # Median latencies
                "median_ttft_ms": dp.median_ttft_ms,
                "median_tpot_ms": dp.median_tpot_ms,
                "median_itl_ms": dp.median_itl_ms,
                "median_e2el_ms": dp.median_e2el_ms,
                # P99 latencies
                "p99_ttft_ms": dp.p99_ttft_ms,
                "p99_tpot_ms": dp.p99_tpot_ms,
                "p99_itl_ms": dp.p99_itl_ms,
                "p99_e2el_ms": dp.p99_e2el_ms,
                # Token counts
                "total_input_tokens": dp.total_input_tokens,
                "total_output_tokens": dp.total_output_tokens,
                # Run metadata
                "duration": dp.duration,
                "completed": dp.completed,
                "num_prompts": dp.num_prompts,
            }

            # Add node metrics summary
            if include_node_metrics and dp.node_metrics:
                node_summary = self._summarize_node_metrics(dp.node_metrics)
                row.update(node_summary)

            rows.append(row)

        return pd.DataFrame(rows)

    def _summarize_node_metrics(self, node_metrics: list[NodeMetrics]) -> dict[str, Any]:
        """Summarize node metrics for DataFrame inclusion.

        Args:
            node_metrics: List of NodeMetrics objects

        Returns:
            Dict with summary statistics
        """
        summary: dict[str, Any] = {
            "num_nodes": len(node_metrics),
            "num_prefill_nodes": sum(1 for n in node_metrics if n.is_prefill),
            "num_decode_nodes": sum(1 for n in node_metrics if n.is_decode),
            "total_batches": sum(len(n.batches) for n in node_metrics),
            "total_memory_snapshots": sum(len(n.memory_snapshots) for n in node_metrics),
        }

        # Aggregate batch metrics across all nodes
        all_batches = [b for n in node_metrics for b in n.batches]
        if all_batches:
            summary["avg_batch_size"] = sum(b.batch_size for b in all_batches) / len(all_batches)
            summary["avg_running_reqs"] = sum(b.running_reqs for b in all_batches) / len(all_batches)
            summary["avg_pending_reqs"] = sum(b.pending_reqs for b in all_batches) / len(all_batches)

        # Aggregate memory metrics across all nodes
        all_memory = [m for n in node_metrics for m in n.memory_snapshots]
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
        prefill_nodes = [n for n in node_metrics if n.is_prefill]
        if prefill_nodes and prefill_nodes[0].config:
            config = prefill_nodes[0].config
            # Handle both NodeConfig dataclass and legacy dict
            if hasattr(config, "server_args"):
                # NodeConfig dataclass - server_args contains flattened config
                server_args = config.server_args or {}
                summary.update({
                    f"prefill_trtllm_config_{k}": v for k, v in server_args.items()
                })
            else:
                # Legacy dict format
                if "trtllm_config" in config:
                    summary.update({
                        f"prefill_trtllm_config_{k}": v for k, v in config["trtllm_config"].items()
                        if k != "command"
                    })
            
            # commands. concatenate all commands from all prefill nodes
            prefill_commands = []
            for n in prefill_nodes:
                if n.config and n.config.server_args and n.config.server_args.get("command"):
                    prefill_commands.append(f"{n.node_name}: {n.config.server_args['command']}")
            summary["prefill_command"] = "\n".join(prefill_commands)

        # Extract decode config (from first decode node - same for all)
        decode_nodes = [n for n in node_metrics if n.is_decode]
        if decode_nodes and decode_nodes[0].config:
            config = decode_nodes[0].config
            # Handle both NodeConfig dataclass and legacy dict
            if hasattr(config, "server_args"):
                # NodeConfig dataclass - server_args contains flattened config
                server_args = config.server_args or {}
                summary.update({
                    f"decode_trtllm_config_{k}": v for k, v in server_args.items()
                    if k != "command"
                })
            else:
                # Legacy dict format
                if "trtllm_config" in config:
                    summary.update({
                        f"decode_trtllm_config_{k}": v for k, v in config["trtllm_config"].items()
                    })
                    
            # commands. concatenate all commands from all decode nodes
            decode_commands = []
            for n in decode_nodes:
                if n.config and n.config.server_args and n.config.server_args.get("command"):
                    decode_commands.append(f"{n.node_name}: {n.config.server_args['command']}")
            summary["decode_command"] = "\n".join(decode_commands)

        return summary

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


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

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

    args = parser.parse_args()

    if args.clear_cache:
        clear_cache(args.log_dir)
        print(f"✅ Cache cleared for {args.log_dir}")

    # Use ResultsExporter in standalone mode
    exporter = ResultsExporter.from_log_dir(args.log_dir)
    exporter.collect_results()

    if not exporter.datapoints:
        print("No datapoints found")
        sys.exit(1)

    print(f"Found {len(exporter.datapoints)} datapoints")

    # Export to CSV and Parquet
    exported = exporter.export()

    for fmt, path in exported.items():
        print(f"✅ Exported {fmt.upper()}: {path}")
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Results exporter for benchmark runs.

Rolls up benchmark datapoints into CSV and Parquet formats after experiment completion.
Parquet is preferred for web transmission due to better compression.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
        exporter = ResultsExporter(config, runtime)
        exporter.export()
    """

    config: "SrtConfig"
    runtime: "RuntimeContext"
    datapoints: list[BenchmarkDatapoint] = field(default_factory=list)

    def collect_results(self) -> list[BenchmarkDatapoint]:
        """Collect benchmark results from the log directory.

        Parses JSON result files from benchmark output directories.

        Returns:
            List of BenchmarkDatapoint objects
        """
        log_dir = self.runtime.log_dir
        benchmark_type = self.config.benchmark.type

        # Handle profiling mode
        if self.config.profiling.enabled:
            benchmark_type = "profiling"

        # Find benchmark result directories
        result_dirs = self._find_result_directories(log_dir, benchmark_type)

        if not result_dirs:
            logger.warning(f"No benchmark result directories found in {log_dir}")
            return []

        datapoints = []
        for result_dir in result_dirs:
            points = self._parse_result_directory(result_dir)
            datapoints.extend(points)

        self.datapoints = datapoints
        logger.info(f"Collected {len(datapoints)} datapoints from {len(result_dirs)} result directories")
        return datapoints

    def _find_result_directories(self, log_dir: Path, benchmark_type: str) -> list[Path]:
        """Find benchmark result directories in the log directory.

        Looks for directories matching patterns like:
        - sa-bench_isl_1024_osl_1024/
        - vllm_isl_1024_osl_1024/

        Args:
            log_dir: Path to the log directory
            benchmark_type: Type of benchmark (e.g., "sa-bench")

        Returns:
            List of paths to result directories
        """
        result_dirs = []

        # Check both log_dir and log_dir/logs
        search_paths = [log_dir]
        logs_subdir = log_dir / "logs"
        if logs_subdir.exists():
            search_paths.append(logs_subdir)

        # Pattern to match benchmark result directories
        isl = self.config.benchmark.isl
        osl = self.config.benchmark.osl

        if isl is not None and osl is not None:
            pattern = re.compile(rf"{re.escape(benchmark_type)}_isl_{isl}_osl_{osl}")

            for search_path in search_paths:
                if not search_path.exists():
                    continue

                for entry in os.listdir(search_path):
                    if pattern.match(entry):
                        full_path = search_path / entry
                        if full_path.is_dir():
                            result_dirs.append(full_path)

        return result_dirs

    def _parse_result_directory(self, result_dir: Path) -> list[BenchmarkDatapoint]:
        """Parse all JSON result files in a directory.

        Args:
            result_dir: Path to directory containing JSON result files

        Returns:
            List of BenchmarkDatapoint objects
        """
        datapoints = []
        run_date = datetime.now().strftime("%Y%m%d_%H%M%S")

        r = self.config.resources

        # Calculate total GPUs
        if r.is_disaggregated:
            total_gpus = (r.prefill_nodes or 0) * r.gpus_per_node + (r.decode_nodes or 0) * r.gpus_per_node
        else:
            total_gpus = (r.agg_nodes or 1) * r.gpus_per_node

        for file in os.listdir(result_dir):
            if not file.endswith(".json"):
                continue

            # Skip pytorch profiler files
            if ".pytorch.json" in file:
                continue

            filepath = result_dir / file
            try:
                with open(filepath) as f:
                    data = json.load(f)

                datapoint = BenchmarkDatapoint(
                    # Run metadata
                    job_id=self.runtime.job_id,
                    run_name=self.runtime.run_name,
                    run_date=run_date,
                    benchmark_type=self.config.benchmark.type,
                    # Resource config
                    gpu_type=r.gpu_type,
                    gpus_per_node=r.gpus_per_node,
                    prefill_nodes=r.prefill_nodes or 0,
                    decode_nodes=r.decode_nodes or 0,
                    prefill_workers=r.num_prefill,
                    decode_workers=r.num_decode,
                    agg_workers=r.num_agg,
                    total_gpus=total_gpus,
                    # Benchmark params
                    isl=self.config.benchmark.isl or 0,
                    osl=self.config.benchmark.osl or 0,
                    concurrency=data.get("max_concurrency", 0),
                    request_rate=data.get("request_rate"),
                    # Throughput metrics
                    output_tps=data.get("output_throughput", 0),
                    total_tps=data.get("total_token_throughput"),
                    request_throughput=data.get("request_throughput"),
                    request_goodput=data.get("request_goodput"),
                    # Mean latencies
                    mean_ttft_ms=data.get("mean_ttft_ms"),
                    mean_tpot_ms=data.get("mean_tpot_ms"),
                    mean_itl_ms=data.get("mean_itl_ms"),
                    mean_e2el_ms=data.get("mean_e2el_ms"),
                    # Median latencies
                    median_ttft_ms=data.get("median_ttft_ms"),
                    median_tpot_ms=data.get("median_tpot_ms"),
                    median_itl_ms=data.get("median_itl_ms"),
                    median_e2el_ms=data.get("median_e2el_ms"),
                    # P99 latencies
                    p99_ttft_ms=data.get("p99_ttft_ms"),
                    p99_tpot_ms=data.get("p99_tpot_ms"),
                    p99_itl_ms=data.get("p99_itl_ms"),
                    p99_e2el_ms=data.get("p99_e2el_ms"),
                    # Token counts
                    total_input_tokens=data.get("total_input_tokens"),
                    total_output_tokens=data.get("total_output_tokens"),
                    # Run metadata
                    duration=data.get("duration"),
                    completed=data.get("completed"),
                    num_prompts=data.get("num_prompts"),
                )
                datapoints.append(datapoint)

            except Exception as e:
                logger.warning(f"Error parsing {filepath}: {e}")
                continue

        # Sort by concurrency
        datapoints.sort(key=lambda x: x.concurrency)
        return datapoints

    def to_dataframe(self):
        """Convert datapoints to pandas DataFrame.

        Returns:
            pandas DataFrame with all datapoints
        """
        import pandas as pd

        if not self.datapoints:
            self.collect_results()

        if not self.datapoints:
            return pd.DataFrame()

        # Convert datapoints to dicts
        rows = []
        for dp in self.datapoints:
            rows.append(
                {
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
            )

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
            output_dir = self.runtime.log_dir

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe()

        if df.empty:
            logger.warning("No datapoints to export")
            return {}

        # Generate filenames
        job_id = self.runtime.job_id
        csv_path = output_dir / f"{job_id}_results.csv"
        parquet_path = output_dir / f"{job_id}_results.parquet"

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
            "job_id": self.runtime.job_id,
            "run_name": self.runtime.run_name,
            "benchmark_type": self.config.benchmark.type,
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

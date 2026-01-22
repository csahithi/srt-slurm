# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Rollup stage mixin for SweepOrchestrator.

Aggregates experiment data from multiple benchmark runs into a single consolidated summary.
Includes node-level metrics parsed from prefill/decode .out and .err files using analysis.srtlog.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from analysis.srtlog.models import NodeMetrics

    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig
    from srtctl.core.topology import Endpoint

logger = logging.getLogger(__name__)


@dataclass
class LaunchCommandRollup:
    """Parsed launch command information for a worker or benchmark."""

    raw_command: str
    command_type: str  # "worker" or "benchmark"

    # Common fields
    model_path: str | None = None
    served_model_name: str | None = None

    # Worker-specific fields
    worker_type: str | None = None  # prefill, decode, agg
    backend_type: str | None = None
    disaggregation_mode: str | None = None
    tp_size: int | None = None
    dp_size: int | None = None
    ep_size: int | None = None
    port: int | None = None
    max_num_seqs: int | None = None
    max_model_len: int | None = None

    # Benchmark-specific fields
    benchmark_type: str | None = None
    base_url: str | None = None
    max_concurrency: int | None = None
    num_prompts: int | None = None
    input_len: int | None = None
    output_len: int | None = None


@dataclass
class NodeRollup:
    """Summary of metrics for a single worker node.

    Derived from analysis.srtlog.models.NodeMetrics with aggregated statistics.
    """

    node_name: str
    worker_type: str  # "prefill", "decode", or "agg"
    worker_id: str

    # Configuration (from NodeMetrics.config)
    tp_size: int | None = None
    dp_size: int | None = None
    ep_size: int | None = None

    # Launch command (parsed from log)
    launch_command: LaunchCommandRollup | None = None

    # Memory metrics (from NodeMetrics.memory_snapshots)
    avail_mem_gb: float | None = None
    mem_usage_gb: float | None = None
    kv_cache_gb: float | None = None
    kv_tokens: int | None = None

    # Batch statistics (aggregated from NodeMetrics.batches)
    total_batches: int = 0
    total_prefill_batches: int = 0
    total_decode_batches: int = 0

    # Prefill-specific stats (also used by agg workers)
    total_new_tokens: int | None = None
    total_cached_tokens: int | None = None
    cache_hit_rate: float | None = None  # Percentage
    avg_input_throughput: float | None = None  # tokens/s
    max_input_throughput: float | None = None  # tokens/s

    # Decode-specific stats (also used by agg workers)
    avg_running_requests: float | None = None
    max_running_requests: int | None = None
    avg_gen_throughput: float | None = None  # tokens/s
    max_gen_throughput: float | None = None  # tokens/s

    # Queue stats
    max_queue_requests: int | None = None
    max_inflight_requests: int | None = None
    max_transfer_requests: int | None = None

    @property
    def is_agg(self) -> bool:
        """Check if this is an aggregated worker."""
        return self.worker_type == "agg"

    @classmethod
    def from_node_metrics(cls, node: "NodeMetrics") -> "NodeRollup":
        """Create NodeRollup from analysis.srtlog.models.NodeMetrics.

        Args:
            node: NodeMetrics object from NodeAnalyzer

        Returns:
            NodeRollup with aggregated statistics
        """
        worker_type = node.node_info.get("worker_type", "unknown")

        rollup = cls(
            node_name=node.node_info.get("node", "unknown"),
            worker_type=worker_type,
            worker_id=node.node_info.get("worker_id", ""),
            tp_size=node.config.get("tp_size"),
            dp_size=node.config.get("dp_size"),
            ep_size=node.config.get("ep_size"),
            total_batches=len(node.batches),
        )

        # Extract memory metrics - aggregate best values from all snapshots
        if node.memory_snapshots:
            # Find best values across all snapshots (some may be partial)
            for mem in node.memory_snapshots:
                if mem.avail_mem_gb is not None and rollup.avail_mem_gb is None:
                    rollup.avail_mem_gb = mem.avail_mem_gb
                if mem.mem_usage_gb is not None and rollup.mem_usage_gb is None:
                    rollup.mem_usage_gb = mem.mem_usage_gb
                if mem.kv_cache_gb is not None:
                    # Take the max kv_cache seen (or sum for multiple allocations)
                    if rollup.kv_cache_gb is None:
                        rollup.kv_cache_gb = mem.kv_cache_gb
                    else:
                        rollup.kv_cache_gb = max(rollup.kv_cache_gb, mem.kv_cache_gb)
                if mem.kv_tokens is not None:
                    # Take the max kv_tokens
                    if rollup.kv_tokens is None:
                        rollup.kv_tokens = mem.kv_tokens
                    else:
                        rollup.kv_tokens = max(rollup.kv_tokens, mem.kv_tokens)

        # Aggregate batch metrics based on worker type
        if node.batches:
            # Check if we have mixed batch types (e.g., TRTLLM decode workers have both)
            batch_types = {b.batch_type for b in node.batches}
            has_mixed = "prefill" in batch_types and "decode" in batch_types

            if worker_type == "agg" or has_mixed:
                # Agg workers or workers with mixed batches need full aggregation
                rollup._aggregate_agg_batches(node.batches)
            elif node.is_prefill:
                rollup._aggregate_prefill_batches(node.batches)
            elif node.is_decode:
                rollup._aggregate_decode_batches(node.batches)

        return rollup

    def _aggregate_prefill_batches(self, batches: list) -> None:
        """Aggregate prefill batch metrics."""
        self.total_prefill_batches = len(batches)

        new_tokens = []
        cached_tokens = []
        input_throughputs = []
        queue_reqs = []
        inflight_reqs = []

        for batch in batches:
            if batch.new_token is not None:
                new_tokens.append(batch.new_token)
            if batch.cached_token is not None:
                cached_tokens.append(batch.cached_token)
            if batch.input_throughput is not None:
                input_throughputs.append(batch.input_throughput)
            if batch.queue_req is not None:
                queue_reqs.append(batch.queue_req)
            if batch.inflight_req is not None:
                inflight_reqs.append(batch.inflight_req)

        if new_tokens:
            self.total_new_tokens = sum(new_tokens)
        if cached_tokens:
            self.total_cached_tokens = sum(cached_tokens)

        # Compute cache hit rate
        if self.total_new_tokens is not None and self.total_cached_tokens is not None:
            total = self.total_new_tokens + self.total_cached_tokens
            if total > 0:
                self.cache_hit_rate = (self.total_cached_tokens / total) * 100

        if input_throughputs:
            self.avg_input_throughput = sum(input_throughputs) / len(input_throughputs)
            self.max_input_throughput = max(input_throughputs)

        if queue_reqs:
            self.max_queue_requests = max(queue_reqs)
        if inflight_reqs:
            self.max_inflight_requests = max(inflight_reqs)

    def _aggregate_decode_batches(self, batches: list) -> None:
        """Aggregate decode batch metrics."""
        self.total_decode_batches = len(batches)

        running_reqs = []
        gen_throughputs = []
        queue_reqs = []
        transfer_reqs = []

        for batch in batches:
            if batch.running_req is not None:
                running_reqs.append(batch.running_req)
            if batch.gen_throughput is not None:
                gen_throughputs.append(batch.gen_throughput)
            if batch.queue_req is not None:
                queue_reqs.append(batch.queue_req)
            if batch.transfer_req is not None:
                transfer_reqs.append(batch.transfer_req)

        if running_reqs:
            self.avg_running_requests = sum(running_reqs) / len(running_reqs)
            self.max_running_requests = max(running_reqs)

        if gen_throughputs:
            self.avg_gen_throughput = sum(gen_throughputs) / len(gen_throughputs)
            self.max_gen_throughput = max(gen_throughputs)

        if queue_reqs:
            self.max_queue_requests = max(queue_reqs)
        if transfer_reqs:
            self.max_transfer_requests = max(transfer_reqs)

    def _aggregate_agg_batches(self, batches: list) -> None:
        """Aggregate metrics for agg workers (handles both prefill and decode batches)."""
        # Separate prefill and decode batches
        prefill_batches = [b for b in batches if b.batch_type == "prefill"]
        decode_batches = [b for b in batches if b.batch_type == "decode"]

        self.total_prefill_batches = len(prefill_batches)
        self.total_decode_batches = len(decode_batches)

        # Aggregate prefill metrics
        if prefill_batches:
            new_tokens = []
            cached_tokens = []
            input_throughputs = []
            inflight_reqs = []

            for batch in prefill_batches:
                if batch.new_token is not None:
                    new_tokens.append(batch.new_token)
                if batch.cached_token is not None:
                    cached_tokens.append(batch.cached_token)
                if batch.input_throughput is not None:
                    input_throughputs.append(batch.input_throughput)
                if batch.inflight_req is not None:
                    inflight_reqs.append(batch.inflight_req)

            if new_tokens:
                self.total_new_tokens = sum(new_tokens)
            if cached_tokens:
                self.total_cached_tokens = sum(cached_tokens)

            # Compute cache hit rate
            if self.total_new_tokens is not None and self.total_cached_tokens is not None:
                total = self.total_new_tokens + self.total_cached_tokens
                if total > 0:
                    self.cache_hit_rate = (self.total_cached_tokens / total) * 100

            if input_throughputs:
                self.avg_input_throughput = sum(input_throughputs) / len(input_throughputs)
                self.max_input_throughput = max(input_throughputs)

            if inflight_reqs:
                self.max_inflight_requests = max(inflight_reqs)

        # Aggregate decode metrics
        if decode_batches:
            running_reqs = []
            gen_throughputs = []
            queue_reqs = []
            transfer_reqs = []

            for batch in decode_batches:
                if batch.running_req is not None:
                    running_reqs.append(batch.running_req)
                if batch.gen_throughput is not None:
                    gen_throughputs.append(batch.gen_throughput)
                if batch.queue_req is not None:
                    queue_reqs.append(batch.queue_req)
                if batch.transfer_req is not None:
                    transfer_reqs.append(batch.transfer_req)

            if running_reqs:
                self.avg_running_requests = sum(running_reqs) / len(running_reqs)
                self.max_running_requests = max(running_reqs)

            if gen_throughputs:
                self.avg_gen_throughput = sum(gen_throughputs) / len(gen_throughputs)
                self.max_gen_throughput = max(gen_throughputs)

            if queue_reqs:
                self.max_queue_requests = max(queue_reqs)
            if transfer_reqs:
                self.max_transfer_requests = max(transfer_reqs)


@dataclass
class NodesSummary:
    """Summary of all worker nodes in the experiment."""

    # Counts
    total_prefill_nodes: int = 0
    total_decode_nodes: int = 0
    total_agg_nodes: int = 0

    # Aggregated prefill stats (from prefill + agg nodes)
    total_prefill_tokens: int | None = None
    total_cached_tokens: int | None = None
    overall_cache_hit_rate: float | None = None  # Percentage
    avg_prefill_input_throughput: float | None = None  # tokens/s per node
    max_prefill_input_throughput: float | None = None  # tokens/s peak

    # Aggregated decode stats (from decode + agg nodes)
    avg_decode_gen_throughput: float | None = None  # tokens/s per node
    max_decode_gen_throughput: float | None = None  # tokens/s peak

    # Memory summary
    total_kv_cache_gb: float | None = None

    # Per-node details
    nodes: list[NodeRollup] = field(default_factory=list)

    @classmethod
    def from_node_metrics_list(cls, nodes: list["NodeMetrics"]) -> "NodesSummary":
        """Create NodesSummary from a list of NodeMetrics.

        Args:
            nodes: List of NodeMetrics from NodeAnalyzer.parse_run_logs()

        Returns:
            NodesSummary with aggregated statistics
        """
        summary = cls()

        # Convert each NodeMetrics to NodeRollup
        for node in nodes:
            rollup = NodeRollup.from_node_metrics(node)
            summary.nodes.append(rollup)

            worker_type = node.node_info.get("worker_type", "unknown")
            if worker_type == "agg":
                summary.total_agg_nodes += 1
            elif node.is_prefill:
                summary.total_prefill_nodes += 1
            elif node.is_decode:
                summary.total_decode_nodes += 1

        # Aggregate across all nodes
        summary._compute_aggregate_stats()

        return summary

    def _compute_aggregate_stats(self) -> None:
        """Compute aggregate statistics across all nodes."""
        # Prefill aggregation (includes both prefill and agg nodes)
        prefill_capable_nodes = [n for n in self.nodes if n.worker_type in ("prefill", "agg")]
        if prefill_capable_nodes:
            total_new = sum(n.total_new_tokens or 0 for n in prefill_capable_nodes)
            total_cached = sum(n.total_cached_tokens or 0 for n in prefill_capable_nodes)

            if total_new > 0 or total_cached > 0:
                self.total_prefill_tokens = total_new
                self.total_cached_tokens = total_cached
                total = total_new + total_cached
                if total > 0:
                    self.overall_cache_hit_rate = (total_cached / total) * 100

            throughputs = [n.avg_input_throughput for n in prefill_capable_nodes if n.avg_input_throughput]
            if throughputs:
                self.avg_prefill_input_throughput = sum(throughputs) / len(throughputs)

            max_throughputs = [n.max_input_throughput for n in prefill_capable_nodes if n.max_input_throughput]
            if max_throughputs:
                self.max_prefill_input_throughput = max(max_throughputs)

        # Decode aggregation (includes both decode and agg nodes)
        decode_capable_nodes = [n for n in self.nodes if n.worker_type in ("decode", "agg")]
        if decode_capable_nodes:
            throughputs = [n.avg_gen_throughput for n in decode_capable_nodes if n.avg_gen_throughput]
            if throughputs:
                self.avg_decode_gen_throughput = sum(throughputs) / len(throughputs)

            max_throughputs = [n.max_gen_throughput for n in decode_capable_nodes if n.max_gen_throughput]
            if max_throughputs:
                self.max_decode_gen_throughput = max(max_throughputs)

        # Memory aggregation
        kv_caches = [n.kv_cache_gb for n in self.nodes if n.kv_cache_gb]
        if kv_caches:
            self.total_kv_cache_gb = sum(kv_caches)


@dataclass
class RollupResult:
    """Consolidated benchmark result for a single concurrency level."""

    concurrency: int
    output_tps: float
    total_tps: float | None = None
    request_throughput: float | None = None
    request_goodput: float | None = None
    request_rate: float | str | None = None

    # Mean latencies
    mean_ttft_ms: float | None = None
    mean_tpot_ms: float | None = None
    mean_itl_ms: float | None = None
    mean_e2el_ms: float | None = None

    # Median latencies
    median_ttft_ms: float | None = None
    median_tpot_ms: float | None = None
    median_itl_ms: float | None = None
    median_e2el_ms: float | None = None

    # P99 latencies
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


@dataclass
class EnvironmentConfig:
    """Environment variables and engine configuration for prefill/decode/agg workers."""

    # Environment variables from config.yaml
    prefill_environment: dict[str, str] = field(default_factory=dict)
    decode_environment: dict[str, str] = field(default_factory=dict)
    aggregated_environment: dict[str, str] = field(default_factory=dict)

    # Engine config from YAML files (TRTLLM) or parsed from logs
    prefill_engine_config: dict[str, Any] = field(default_factory=dict)
    decode_engine_config: dict[str, Any] = field(default_factory=dict)
    aggregated_engine_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class RollupSummary:
    """Complete rollup summary for an experiment."""

    # Experiment identification
    job_id: str
    job_name: str
    generated_at: str

    # Configuration
    model_path: str
    model_name: str
    precision: str
    gpu_type: str
    gpus_per_node: int
    backend_type: str
    frontend_type: str

    # Resource allocation
    is_disaggregated: bool
    total_nodes: int
    total_gpus: int
    prefill_nodes: int | None = None
    decode_nodes: int | None = None
    prefill_workers: int | None = None
    decode_workers: int | None = None
    prefill_gpus: int | None = None
    decode_gpus: int | None = None
    agg_nodes: int | None = None
    agg_workers: int | None = None

    # Benchmark configuration
    benchmark_type: str = ""
    isl: int | None = None
    osl: int | None = None
    concurrencies: list[int] = field(default_factory=list)

    # Aggregated results
    results: list[RollupResult] = field(default_factory=list)

    # Summary statistics (computed from results)
    max_output_tps: float | None = None
    max_total_tps: float | None = None
    min_mean_ttft_ms: float | None = None
    min_mean_itl_ms: float | None = None

    # Node-level metrics
    nodes_summary: NodesSummary | None = None

    # Environment and engine configuration
    environment_config: EnvironmentConfig | None = None

    # Launch commands
    benchmark_command: LaunchCommandRollup | None = None

    # Tags
    tags: list[str] = field(default_factory=list)

    def compute_summary_stats(self) -> None:
        """Compute summary statistics from results."""
        if not self.results:
            return

        output_tps_values = [r.output_tps for r in self.results if r.output_tps is not None]
        total_tps_values = [r.total_tps for r in self.results if r.total_tps is not None]
        ttft_values = [r.mean_ttft_ms for r in self.results if r.mean_ttft_ms is not None]
        itl_values = [r.mean_itl_ms for r in self.results if r.mean_itl_ms is not None]

        if output_tps_values:
            self.max_output_tps = max(output_tps_values)
        if total_tps_values:
            self.max_total_tps = max(total_tps_values)
        if ttft_values:
            self.min_mean_ttft_ms = min(ttft_values)
        if itl_values:
            self.min_mean_itl_ms = min(itl_values)


class RollupStageMixin:
    """Mixin for rollup stage that consolidates experiment data.

    Requires:
        self.config: SrtConfig
        self.runtime: RuntimeContext
        self.endpoints: list[Endpoint]
    """

    # Type hints for mixin dependencies
    config: "SrtConfig"
    runtime: "RuntimeContext"

    @property
    def endpoints(self) -> list["Endpoint"]:
        """Endpoint allocation topology."""
        ...

    def run_rollup(self, tags: list[str] | None = None) -> Path | None:
        """Run the rollup stage to consolidate experiment data.

        Args:
            tags: Optional list of tags for the experiment

        Returns:
            Path to the generated rollup.json file, or None if rollup failed
        """
        logger.info("Running rollup stage")

        try:
            # Collect benchmark results
            results = self._collect_benchmark_results()

            if not results:
                logger.warning("No benchmark results found to rollup")
                return None

            # Collect node metrics using analysis.srtlog
            nodes_summary = self._collect_node_metrics()

            # Collect benchmark launch command
            benchmark_command = self._collect_benchmark_command()

            # Collect environment and engine configuration
            environment_config = self._collect_environment_config()

            # Build rollup summary
            summary = self._build_rollup_summary(results, tags, nodes_summary, benchmark_command, environment_config)

            # Write rollup.json
            rollup_path = self.runtime.log_dir / "rollup.json"
            self._write_rollup(summary, rollup_path)

            logger.info("Rollup complete: %s", rollup_path)
            logger.info(
                "Summary: %d results, max output TPS: %.2f, %d nodes",
                len(summary.results),
                summary.max_output_tps or 0,
                len(nodes_summary.nodes) if nodes_summary else 0,
            )

            return rollup_path

        except Exception as e:
            logger.error("Rollup failed: %s", e)
            return None

    def _collect_benchmark_results(self) -> list[dict[str, Any]]:
        """Collect all benchmark result JSON files from the log directory.

        Uses the appropriate benchmark parser based on config.benchmark.type.

        Returns:
            List of parsed benchmark result dicts
        """
        results = []
        benchmark_type = self.config.benchmark.type

        try:
            from analysis.srtlog.parsers import get_benchmark_parser, list_benchmark_parsers

            # Get the appropriate parser
            try:
                parser = get_benchmark_parser(benchmark_type)
                logger.debug("Using %s benchmark parser", benchmark_type)
            except ValueError:
                logger.warning(
                    "No parser for benchmark type '%s', available: %s. Using fallback.",
                    benchmark_type,
                    list_benchmark_parsers(),
                )
                parser = None

            # Try parser-specific result collection first
            if parser is not None:
                # For mooncake-router, look for AIPerf results
                if hasattr(parser, "find_aiperf_results"):
                    aiperf_files = parser.find_aiperf_results(self.runtime.log_dir)
                    for aiperf_path in aiperf_files:
                        result = parser.parse_result_json(aiperf_path)
                        if result.get("output_tps") is not None:
                            results.append(result)
                            logger.debug("Loaded AIPerf result: %s", aiperf_path)

                # For sa-bench style, look for result directories
                if hasattr(parser, "parse_result_directory"):
                    for entry in self.runtime.log_dir.iterdir():
                        if not entry.is_dir():
                            continue
                        # Match patterns like sa-bench_isl_X_osl_Y
                        if "_isl_" in entry.name and "_osl_" in entry.name:
                            logger.debug("Found benchmark results directory: %s", entry.name)
                            dir_results = parser.parse_result_directory(entry)
                            results.extend(dir_results)

        except ImportError:
            logger.debug("analysis.srtlog.parsers not available, using fallback")
            parser = None

        # Fallback: direct JSON parsing
        if not results:
            for entry in self.runtime.log_dir.iterdir():
                if not entry.is_dir():
                    continue

                # Match patterns like sa-bench_isl_X_osl_Y, vllm_isl_X_osl_Y
                if "_isl_" in entry.name and "_osl_" in entry.name:
                    logger.debug("Found benchmark results directory: %s", entry.name)

                    # Parse all JSON files in the directory
                    for json_file in entry.glob("*.json"):
                        try:
                            with open(json_file) as f:
                                data = json.load(f)
                                results.append(data)
                                logger.debug("Loaded result: %s", json_file.name)
                        except Exception as e:
                            logger.warning("Failed to parse %s: %s", json_file, e)

        # Sort by concurrency
        results.sort(key=lambda x: x.get("max_concurrency", 0) or 0)

        logger.info("Collected %d benchmark results", len(results))
        return results

    def _collect_node_metrics(self) -> NodesSummary | None:
        """Collect node metrics from prefill/decode log files.

        Uses the appropriate node parser based on config.backend_type.
        Falls back through parser versions if needed (e.g., sglang -> sglang-v2).

        Returns:
            NodesSummary with aggregated node statistics, or None if parsing fails
        """
        backend_type = self.config.backend_type
        log_dir = self.runtime.log_dir

        try:
            from analysis.srtlog.parsers import get_node_parser, list_node_parsers

            # Try parsers in order of preference
            parser_order = self._get_parser_order(backend_type)
            logger.debug("Parser order for %s: %s", backend_type, parser_order)

            nodes = []
            used_parser = None
            parser = None

            for parser_type in parser_order:
                try:
                    parser = get_node_parser(parser_type)
                    nodes = parser.parse_logs(log_dir)

                    # Check if we got meaningful results (batches or config)
                    total_batches = sum(len(n.batches) for n in nodes)
                    has_config = any(n.config for n in nodes)
                    if total_batches > 0 or has_config:
                        used_parser = parser_type
                        logger.info("Using %s parser: found %d nodes with %d batches", parser_type, len(nodes), total_batches)
                        break
                    else:
                        logger.debug("%s parser found no batches, trying next", parser_type)

                except ValueError:
                    logger.debug("Parser %s not available", parser_type)
                    continue

            if not nodes:
                logger.warning("No node metrics found in %s with any parser", log_dir)
                return None

            # Build summary from parsed nodes
            summary = NodesSummary.from_node_metrics_list(nodes)

            # Parse launch commands for each node
            if parser is not None and hasattr(parser, "parse_launch_command"):
                self._add_launch_commands_to_summary(summary, parser, log_dir)

            if summary.total_agg_nodes > 0:
                logger.info("Node summary (%s): %d agg nodes", used_parser, summary.total_agg_nodes)
            else:
                logger.info(
                    "Node summary (%s): %d prefill, %d decode nodes",
                    used_parser,
                    summary.total_prefill_nodes,
                    summary.total_decode_nodes,
                )

            return summary

        except ImportError:
            logger.warning("analysis.srtlog.parsers not available, skipping node metrics")
            return None
        except Exception as e:
            logger.warning("Failed to collect node metrics: %s", e)
            return None

    def _add_launch_commands_to_summary(self, summary: NodesSummary, parser: Any, log_dir: Path) -> None:
        """Parse and add launch commands to each node in the summary.

        Args:
            summary: NodesSummary to update
            parser: Node parser with parse_launch_command method
            log_dir: Directory containing log files
        """
        import os

        for node_rollup in summary.nodes:
            # Find the log file for this node
            node_name = node_rollup.node_name
            worker_type = node_rollup.worker_type
            worker_id = node_rollup.worker_id

            # Try both .out and .err files
            for ext in [".out", ".err"]:
                log_file = log_dir / f"{node_name}_{worker_type}_{worker_id}{ext}"
                if log_file.exists():
                    try:
                        content = log_file.read_text(errors="replace")
                        cmd = parser.parse_launch_command(content, worker_type=worker_type)
                        if cmd:
                            node_rollup.launch_command = LaunchCommandRollup(
                                raw_command=cmd.raw_command,
                                command_type="worker",
                                model_path=cmd.model_path,
                                served_model_name=cmd.served_model_name,
                                worker_type=worker_type,
                                backend_type=cmd.backend_type,
                                disaggregation_mode=cmd.disaggregation_mode,
                                tp_size=cmd.tp_size,
                                dp_size=cmd.dp_size,
                                ep_size=cmd.ep_size,
                                port=cmd.port,
                                max_num_seqs=cmd.max_num_seqs,
                                max_model_len=cmd.max_model_len,
                            )
                            logger.debug("Parsed launch command for %s_%s_%s", node_name, worker_type, worker_id)
                            break
                    except Exception as e:
                        logger.debug("Failed to parse launch command from %s: %s", log_file, e)

    def _collect_benchmark_command(self) -> LaunchCommandRollup | None:
        """Parse the benchmark launch command from benchmark.out.

        Returns:
            LaunchCommandRollup with benchmark parameters, or None if not found
        """
        benchmark_type = self.config.benchmark.type
        log_dir = self.runtime.log_dir

        try:
            from analysis.srtlog.parsers import get_benchmark_parser

            parser = get_benchmark_parser(benchmark_type)

            # Look for benchmark.out file
            benchmark_out = log_dir / "benchmark.out"
            if not benchmark_out.exists():
                logger.debug("benchmark.out not found in %s", log_dir)
                return None

            content = benchmark_out.read_text(errors="replace")
            cmd = parser.parse_launch_command(content)

            if cmd:
                return LaunchCommandRollup(
                    raw_command=cmd.raw_command,
                    command_type="benchmark",
                    model_path=cmd.model,
                    benchmark_type=cmd.benchmark_type,
                    base_url=cmd.base_url,
                    max_concurrency=cmd.max_concurrency,
                    num_prompts=cmd.num_prompts,
                    input_len=cmd.input_len,
                    output_len=cmd.output_len,
                )

        except ImportError:
            logger.debug("analysis.srtlog.parsers not available")
        except ValueError as e:
            logger.debug("No benchmark parser for %s: %s", benchmark_type, e)
        except Exception as e:
            logger.debug("Failed to parse benchmark command: %s", e)

        return None

    def _collect_environment_config(self) -> EnvironmentConfig | None:
        """Collect environment variables and engine config from config files.

        Parses:
        1. config.yaml for prefill_environment and decode_environment
        2. YAML config files (e.g., trtllm_config_prefill.yaml) for engine settings

        Returns:
            EnvironmentConfig with environment variables and engine config, or None if not found
        """
        log_dir = self.runtime.log_dir

        try:
            import yaml
        except ImportError:
            logger.debug("PyYAML not available, skipping environment config collection")
            return None

        config = EnvironmentConfig()

        # Try to find config.yaml in the job output directory
        # It could be in log_dir, log_dir.parent, or a sibling directory
        config_paths = [
            log_dir / "config.yaml",
            log_dir.parent / "config.yaml",
            log_dir.parent.parent / "config.yaml",
        ]

        config_yaml = None
        for path in config_paths:
            if path.exists():
                config_yaml = path
                break

        if config_yaml:
            try:
                with open(config_yaml) as f:
                    job_config = yaml.safe_load(f)

                backend_section = job_config.get("backend", {})

                # Extract environment variables
                if "prefill_environment" in backend_section:
                    config.prefill_environment = backend_section["prefill_environment"]
                    logger.debug("Found prefill_environment with %d vars", len(config.prefill_environment))

                if "decode_environment" in backend_section:
                    config.decode_environment = backend_section["decode_environment"]
                    logger.debug("Found decode_environment with %d vars", len(config.decode_environment))

                if "aggregated_environment" in backend_section:
                    config.aggregated_environment = backend_section["aggregated_environment"]
                    logger.debug("Found aggregated_environment with %d vars", len(config.aggregated_environment))

                # For TRTLLM, also extract inline engine config
                if "trtllm_config" in backend_section:
                    trtllm_config = backend_section["trtllm_config"]
                    if "prefill" in trtllm_config:
                        config.prefill_engine_config = trtllm_config["prefill"]
                    if "decode" in trtllm_config:
                        config.decode_engine_config = trtllm_config["decode"]
                    if "aggregated" in trtllm_config:
                        config.aggregated_engine_config = trtllm_config["aggregated"]

                # For SGLang, extract sglang_config if present
                if "sglang_config" in backend_section:
                    sglang_config = backend_section["sglang_config"]
                    if "prefill" in sglang_config:
                        config.prefill_engine_config = sglang_config["prefill"]
                    if "decode" in sglang_config:
                        config.decode_engine_config = sglang_config["decode"]
                    if "aggregated" in sglang_config:
                        config.aggregated_engine_config = sglang_config["aggregated"]

            except Exception as e:
                logger.debug("Failed to parse config.yaml: %s", e)

        # Also look for separate YAML config files (e.g., trtllm_config_prefill.yaml)
        prefill_yaml = log_dir / "trtllm_config_prefill.yaml"
        decode_yaml = log_dir / "trtllm_config_decode.yaml"

        if prefill_yaml.exists() and not config.prefill_engine_config:
            try:
                with open(prefill_yaml) as f:
                    config.prefill_engine_config = yaml.safe_load(f)
                logger.debug("Loaded prefill engine config from %s", prefill_yaml)
            except Exception as e:
                logger.debug("Failed to parse %s: %s", prefill_yaml, e)

        if decode_yaml.exists() and not config.decode_engine_config:
            try:
                with open(decode_yaml) as f:
                    config.decode_engine_config = yaml.safe_load(f)
                logger.debug("Loaded decode engine config from %s", decode_yaml)
            except Exception as e:
                logger.debug("Failed to parse %s: %s", decode_yaml, e)

        # Return None if we didn't find anything
        if not any([
            config.prefill_environment,
            config.decode_environment,
            config.aggregated_environment,
            config.prefill_engine_config,
            config.decode_engine_config,
            config.aggregated_engine_config,
        ]):
            logger.debug("No environment or engine config found")
            return None

        # Log what we found
        env_counts = []
        if config.prefill_environment:
            env_counts.append(f"{len(config.prefill_environment)} prefill")
        if config.decode_environment:
            env_counts.append(f"{len(config.decode_environment)} decode")
        if config.aggregated_environment:
            env_counts.append(f"{len(config.aggregated_environment)} agg")
        
        if env_counts:
            logger.info("Collected environment vars: %s", ", ".join(env_counts))

        return config

    def _get_parser_order(self, backend_type: str) -> list[str]:
        """Get the order of parsers to try for a given backend type.

        Args:
            backend_type: Backend type from config (e.g., "sglang", "trtllm")

        Returns:
            List of parser types to try in order
        """
        # Map backend types to parser order (try newer formats first)
        parser_orders = {
            "sglang": ["sglang-v2", "sglang"],  # Try v2 first (newer logs)
            "trtllm": ["trtllm"],
        }

        return parser_orders.get(backend_type, [backend_type])

    def _build_rollup_summary(
        self,
        results: list[dict[str, Any]],
        tags: list[str] | None = None,
        nodes_summary: NodesSummary | None = None,
        benchmark_command: LaunchCommandRollup | None = None,
        environment_config: EnvironmentConfig | None = None,
    ) -> RollupSummary:
        """Build a RollupSummary from collected results.

        Args:
            results: List of parsed benchmark result dicts
            tags: Optional tags for the experiment
            nodes_summary: Optional node-level metrics summary
            benchmark_command: Optional parsed benchmark launch command
            environment_config: Optional environment and engine configuration

        Returns:
            RollupSummary instance
        """
        r = self.config.resources
        b = self.config.benchmark

        # Determine topology
        is_disaggregated = r.is_disaggregated

        if is_disaggregated:
            total_gpus = r.prefill_gpus + r.decode_gpus
        else:
            total_gpus = (r.agg_nodes or 1) * r.gpus_per_node

        # Build summary
        summary = RollupSummary(
            # Identification
            job_id=self.runtime.job_id,
            job_name=self.config.name,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # Model config
            model_path=str(self.runtime.model_path),
            model_name=self.config.served_model_name,
            precision=self.config.model.precision,
            gpu_type=r.gpu_type,
            gpus_per_node=r.gpus_per_node,
            backend_type=self.config.backend_type,
            frontend_type=self.config.frontend.type,
            # Resource allocation
            is_disaggregated=is_disaggregated,
            total_nodes=r.total_nodes,
            total_gpus=total_gpus,
            # Benchmark config
            benchmark_type=b.type,
            isl=b.isl,
            osl=b.osl,
            concurrencies=b.get_concurrency_list(),
            # Node metrics
            nodes_summary=nodes_summary,
            # Environment and engine configuration
            environment_config=environment_config,
            # Launch commands
            benchmark_command=benchmark_command,
            # Tags
            tags=tags or [],
        )

        # Add disaggregated-specific fields
        if is_disaggregated:
            summary.prefill_nodes = r.prefill_nodes
            summary.decode_nodes = r.decode_nodes
            summary.prefill_workers = r.num_prefill
            summary.decode_workers = r.num_decode
            summary.prefill_gpus = r.prefill_gpus
            summary.decode_gpus = r.decode_gpus
        else:
            summary.agg_nodes = r.agg_nodes
            summary.agg_workers = r.num_agg

        # Convert results to RollupResult objects
        for data in results:
            result = RollupResult(
                concurrency=data.get("max_concurrency", 0),
                output_tps=data.get("output_throughput", 0),
                total_tps=data.get("total_token_throughput"),
                request_throughput=data.get("request_throughput"),
                request_goodput=data.get("request_goodput"),
                request_rate=data.get("request_rate"),
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
                # Metadata
                duration=data.get("duration"),
                completed=data.get("completed"),
                num_prompts=data.get("num_prompts"),
            )
            summary.results.append(result)

        # Compute summary statistics
        summary.compute_summary_stats()

        return summary

    def _write_rollup(self, summary: RollupSummary, path: Path) -> None:
        """Write rollup summary to JSON file.

        Args:
            summary: RollupSummary to write
            path: Output file path
        """
        # Convert to dict, handling nested dataclasses
        data = asdict(summary)

        # Write with nice formatting
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.debug("Wrote rollup to %s", path)

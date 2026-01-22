# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Rollup dataclasses for experiment data consolidation.

These models represent the structure of rollup.json output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from analysis.srtlog.models import NodeMetrics


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
    def from_node_metrics(cls, node: NodeMetrics) -> NodeRollup:
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
    def from_node_metrics_list(cls, nodes: list[NodeMetrics]) -> NodesSummary:
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


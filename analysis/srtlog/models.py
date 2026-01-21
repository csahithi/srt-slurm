"""
Domain models for benchmark analysis

Centralized location for all data models and type definitions.
Includes both dataclasses (for objects) and TypedDicts (for dict typing).
"""

from dataclasses import dataclass, field, fields
from datetime import datetime
from shutil import register_unpack_format
from typing import Any, Literal

import json
import os
from typing_extensions import override, TypedDict
from pydantic import BaseModel, model_validator
from pathlib import Path

class RunMetadata(BaseModel):
    """
    Metadata about a benchmark run from {jobid}.json.
    this provides a materialized view on disk for the srtctl.core.schema.SrtConfig class.
    """

    job_id: str
    job_name: str
    run_date: str

    # modelconfig
    model_dir: str
    container: str
    precision: str

    # resourceconfig
    gpu_type: str
    gpus_per_node: int = 0
    prefill_nodes: int = 0
    decode_nodes: int = 0
    prefill_workers: int = 0
    decode_workers: int = 0
    agg_nodes: int = 0
    agg_workers: int = 0

    # Slurm configs
    path: Path
    partition: str
    
    # Frontend configs
    frontend_type: str = "dynamo"
    enable_multiple_frontends: bool = False
    num_additional_frontends: int = 0

    @model_validator(mode="after")
    def validate_resource_config(self) -> "RunMetadata":
        """
        Make sure that resource config is correctly parsed.
        1. either disagg fields are set or agg fields are set, but not both.
        """
        disagg_nodes_are_correct = self.prefill_nodes > 0 and self.decode_nodes > 0
        disagg_workers_are_correct = self.prefill_workers + self.decode_workers > 0

        agg_nodes_are_correct = self.agg_nodes > 0
        agg_workers_are_correct = self.agg_workers > 0

        # either disagg_nodes and workers or agg_nodes and workers are set, but not both.
        if disagg_nodes_are_correct and disagg_workers_are_correct:
            if agg_nodes_are_correct or agg_workers_are_correct:
                raise ValueError("Disaggregated mode cannot have agg_nodes and agg_workers set.")

        if agg_nodes_are_correct and agg_workers_are_correct:
            if disagg_nodes_are_correct or disagg_workers_are_correct:
                raise ValueError("Aggregated mode cannot have disagg_nodes and disagg_workers set.")

        return self

    @classmethod
    def from_json(cls, json_data: dict, run_path: str) -> "RunMetadata":
        """Create from {jobid}.json metadata format.

        Supports both v1.0 (run_metadata section) and v2.0 (flat structure) formats.

        Args:
            json_data: Parsed JSON from {jobid}.json file
            run_path: Path to the run directory

        Returns:
            RunMetadata instance
        """
        # Check for v2.0 format (has "version": "2.0" or "orchestrator": true)
        is_v2 = json_data.get("version") == "2.0" or json_data.get("orchestrator")

        if is_v2:
            # v2.0 format: flat structure with resources section
            resources = json_data.get("resources", {})
            model = json_data.get("model", {})
            return cls(
                job_id=str(json_data.get("job_id", "")),
                job_name=json_data.get("job_name", ""),
                run_date=json_data.get("generated_at", ""),
                
                model_dir=model.get("path", ""),
                precision=model.get("precision", ""),
                container=model.get("container", ""),
                
                prefill_nodes=resources.get("prefill_nodes", 0),
                decode_nodes=resources.get("decode_nodes", 0),
                prefill_workers=resources.get("prefill_workers", 0),
                decode_workers=resources.get("decode_workers", 0),
                gpus_per_node=resources.get("gpus_per_node", 0),
                gpu_type=resources.get("gpu_type", ""),
                agg_nodes=resources.get("agg_nodes", 0),
                agg_workers=resources.get("agg_workers", 0),

                frontend_type=json_data.get("frontend_type", "dynamo"),
                enable_multiple_frontends=json_data.get("enable_multiple_frontends", False),
                num_additional_frontends=json_data.get("num_additional_frontends", 0),
                
                path=Path(run_path),
                partition=json_data.get("partition", ""),
            )
        else:
            raise ValueError("Unsupported version of {jobid}.json file.")
            # # v1.0 format: run_metadata section
            # run_meta = json_data.get("run_metadata", {})
            # mode = run_meta.get("mode", "disaggregated")

            # return cls(
            #     job_id=run_meta.get("slurm_job_id", ""),
            #     path=run_path,
            #     run_date=run_meta.get("run_date", ""),
            #     container=run_meta.get("container", ""),
            #     prefill_nodes=run_meta.get("prefill_nodes", 0),
            #     decode_nodes=run_meta.get("decode_nodes", 0),
            #     prefill_workers=run_meta.get("prefill_workers", 0),
            #     decode_workers=run_meta.get("decode_workers", 0),
            #     mode=mode,
            #     job_name=run_meta.get("job_name", ""),
            #     partition=run_meta.get("partition", ""),
            #     model_dir=run_meta.get("model_dir", ""),
            #     gpus_per_node=run_meta.get("gpus_per_node", 0),
            #     gpu_type=run_meta.get("gpu_type", ""),
            #     enable_multiple_frontends=run_meta.get("enable_multiple_frontends", False),
            #     num_additional_frontends=run_meta.get("num_additional_frontends", 0),
            #     agg_nodes=run_meta.get("agg_nodes", 0),
            #     agg_workers=run_meta.get("agg_workers", 0),
            # )

    @property
    def mode(self) -> str:
        """Get the mode of the run."""
        if self.prefill_nodes > 0 and self.decode_nodes > 0:
            return "disaggregated"
        return "aggregated"

    @property
    def is_aggregated(self) -> bool:
        """Check if this is an aggregated mode run."""
        return self.mode == "aggregated" or self.agg_nodes > 0

    @property
    def total_gpus(self) -> int:
        """Calculate total GPU count for both modes."""
        if self.is_aggregated:
            return self.agg_nodes * self.gpus_per_node
        return (self.prefill_nodes + self.decode_nodes) * self.gpus_per_node

    @property
    def topology_label(self) -> str:
        """Get topology label appropriate for the mode.

        Returns:
            "XP/YD" for disaggregated, "XA" for aggregated
        """
        if self.is_aggregated:
            return f"{self.agg_workers}A"
        return f"{self.prefill_workers}P/{self.decode_workers}D"

    @property
    def formatted_date(self) -> str:
        """Get human-readable date string (e.g., 'Nov 10').

        Returns:
            Formatted date string like "Nov 10", or raw date if parsing fails
        """
        try:
            # Parse YYYYMMDD_HHMMSS format
            dt = datetime.strptime(self.run_date, "%Y%m%d_%H%M%S")
            return dt.strftime("%b %d").replace(" 0", " ")
        except (ValueError, TypeError):
            return self.run_date

class ProfilerMetadata(BaseModel):
    """
    
    This provides a materialized view on disk for the srtctl.core.schema.ProfilingConfig class.

    Attributes:
        profiler_type: Type of profiler (e.g., 'sa-bench', 'vllm-bench')
        isl: Input sequence length
        osl: Output sequence length
        concurrencies: Concurrency levels to test (e.g., '1x2x4x8')
        req_rate: Request rate (e.g., 'inf' or numeric)
    """

    profiler_type: Literal["sa-bench", "manual"]
    isl: int
    osl: int
    concurrencies: list[int] | str
    req_rate: str | int = "inf"

    @classmethod
    def from_job_id_json(cls, json_data: dict) -> "ProfilerMetadata":
        """Create from {jobid}.json profiler_metadata section."""
        if json_data.get("version") == "2.0":
            benchmark = json_data.get("benchmark", {})
            return cls(
                profiler_type=benchmark.get("type"),
                isl=benchmark.get("isl"),
                osl=benchmark.get("osl"),
                concurrencies=benchmark.get("concurrencies", []),
                req_rate=benchmark.get("req-rate", "inf"),
            )
        else:
            raise ValueError("Unsupported version of {jobid}.json file.")
    
    

@dataclass
class ProfilerResults:
    """Results from profiler benchmarks.
    Only supports sa-bench for now.

    Parses 32 out of 39 fields from benchmark JSON output.

    NOT PARSED (7 fields):
    - input_lens, output_lens, ttfts, itls: Per-request arrays (too large for in-memory storage)
    - errors, generated_texts: Per-request data (not needed for aggregate analysis)
    - tokenizer_id, best_of, burstiness: Metadata not critical for dashboards
    """
    # Primary throughput metrics (per concurrency level)
    output_tps: list[float] = field(default_factory=list)
    total_tps: list[float] = field(default_factory=list)
    request_throughput: list[float] = field(default_factory=list)
    request_goodput: list[float | None] = field(default_factory=list)
    concurrency_values: list[int] = field(default_factory=list)
    request_rate: list[float] = field(default_factory=list)

    # Latency metrics - mean (per concurrency level)
    mean_ttft_ms: list[float] = field(default_factory=list)
    mean_tpot_ms: list[float] = field(default_factory=list)
    mean_itl_ms: list[float] = field(default_factory=list)
    mean_e2el_ms: list[float] = field(default_factory=list)

    # Latency metrics - median (per concurrency level)
    median_ttft_ms: list[float] = field(default_factory=list)
    median_tpot_ms: list[float] = field(default_factory=list)
    median_itl_ms: list[float] = field(default_factory=list)
    median_e2el_ms: list[float] = field(default_factory=list)

    # Latency metrics - p99 (per concurrency level)
    p99_ttft_ms: list[float] = field(default_factory=list)
    p99_tpot_ms: list[float] = field(default_factory=list)
    p99_itl_ms: list[float] = field(default_factory=list)
    p99_e2el_ms: list[float] = field(default_factory=list)

    # Latency metrics - std dev (per concurrency level)
    std_ttft_ms: list[float] = field(default_factory=list)
    std_tpot_ms: list[float] = field(default_factory=list)
    std_itl_ms: list[float] = field(default_factory=list)
    std_e2el_ms: list[float] = field(default_factory=list)

    # Token counts (per concurrency level)
    total_input_tokens: list[int] = field(default_factory=list)
    total_output_tokens: list[int] = field(default_factory=list)

    # Run metadata (per concurrency level)
    backend: list[str] = field(default_factory=list)
    model_id: list[str] = field(default_factory=list)
    date: list[str] = field(default_factory=list)
    duration: list[float] = field(default_factory=list)
    completed: list[int] = field(default_factory=list)
    num_prompts: list[int] = field(default_factory=list)

    def load_results_from_cached_json(self, results: dict) -> None:
        """Add actual benchmark results from profiler output files.

        Args:
            results: Dict with all benchmark metrics from parsed JSON files
        """
        # Primary metrics
        self.concurrency_values = results.get("concurrencies", [])
        self.output_tps = results.get("output_tps", [])
        self.total_tps = results.get("total_tps", [])
        self.request_throughput = results.get("request_throughput", [])
        self.request_goodput = results.get("request_goodput", [])
        self.request_rate = results.get("request_rate", [])

        # Mean latencies
        self.mean_ttft_ms = results.get("mean_ttft_ms", [])
        self.mean_tpot_ms = results.get("mean_tpot_ms", [])
        self.mean_itl_ms = results.get("mean_itl_ms", [])
        self.mean_e2el_ms = results.get("mean_e2el_ms", [])

        # Median latencies
        self.median_ttft_ms = results.get("median_ttft_ms", [])
        self.median_tpot_ms = results.get("median_tpot_ms", [])
        self.median_itl_ms = results.get("median_itl_ms", [])
        self.median_e2el_ms = results.get("median_e2el_ms", [])

        # P99 latencies
        self.p99_ttft_ms = results.get("p99_ttft_ms", [])
        self.p99_tpot_ms = results.get("p99_tpot_ms", [])
        self.p99_itl_ms = results.get("p99_itl_ms", [])
        self.p99_e2el_ms = results.get("p99_e2el_ms", [])

        # Std dev latencies
        self.std_ttft_ms = results.get("std_ttft_ms", [])
        self.std_tpot_ms = results.get("std_tpot_ms", [])
        self.std_itl_ms = results.get("std_itl_ms", [])
        self.std_e2el_ms = results.get("std_e2el_ms", [])

        # Token counts
        self.total_input_tokens = results.get("total_input_tokens", [])
        self.total_output_tokens = results.get("total_output_tokens", [])

        # Metadata
        self.backend = results.get("backend", [])
        self.model_id = results.get("model_id", [])
        self.date = results.get("date", [])
        self.duration = results.get("duration", [])
        self.completed = results.get("completed", [])
        self.num_prompts = results.get("num_prompts", [])

    def get_datapoint(self, index: int) -> dict[str, Any]:
        """Get a datapoint for a given index by iterating over all fields."""
        
        result: dict[str, Any] = {}
        for f in fields(self):
            values = getattr(self, f.name)
            # Rename concurrency_values -> concurrency for output
            key = "concurrency" if f.name == "concurrency_values" else f.name
            if key == "concurrency":
                default = 0
            else:
                default = None
                
            result[key] = values[index] if index < len(values) else default
        return result

@dataclass
class BenchmarkRun:
    """Complete benchmark run with metadata and profiler results."""

    metadata: RunMetadata
    profiler_metadata: ProfilerMetadata

    profiler_results: ProfilerResults = field(default_factory=ProfilerResults)
    missing_concurrencies: list[int] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_json_file(cls, run_path: str, job_id: int | None = None) -> "BenchmarkRun | None":
        """
        Each run directory contains a {jobid}.json file.

        Args:
            run_path: Path to the run directory containing {jobid}.json
            job_id: Job ID
        Returns:
            BenchmarkRun instance or None if file not found/invalid
        """

        # Extract job ID from directory name
        if job_id is None:
            dirname = os.path.basename(run_path)
            job_id = dirname.split("_")[0]
            if not job_id.isdigit():
                raise ValueError(f"Invalid job ID: {job_id}")
        
        json_path = os.path.join(run_path, f"{job_id}.json")

        if not os.path.exists(json_path):
            return None

        try:
            with open(json_path) as f:
                json_data = json.load(f)

            metadata = RunMetadata.from_json(json_data, run_path)
            profiler_metadata = ProfilerMetadata.from_job_id_json(json_data)
            tags = json_data.get("tags", [])

            return cls(metadata=metadata, profiler_metadata=profiler_metadata, tags=tags)
        except Exception:
            return None

    @property
    def job_id(self) -> str:
        """Convenience property for job ID."""
        return self.metadata.job_id

    @property
    def total_gpus(self) -> int:
        """Calculate total GPU count."""
        return self.metadata.total_gpus

    def check_completeness(self) -> None:
        """Check if all expected benchmark results are present.

        Compares expected concurrencies from profiler metadata with actual results.
        Updates is_complete and missing_concurrencies fields.
        """
        # Parse expected concurrencies from metadata
        if not self.profiler_metadata.concurrencies:
            # No expected concurrencies specified, assume manual run
            self.missing_concurrencies = []
            return

        expected = set()
        for val in self.profiler_metadata.concurrencies.split("x"):
            try:
                expected.add(int(val.strip()))
            except ValueError:
                continue

        # Get actual concurrencies from results
        actual = set(self.profiler_results.concurrency_values)

        # Find missing ones
        missing = expected - actual

        self.missing_concurrencies = sorted(missing)

    @property
    def is_complete(self) -> bool:
        """Check if all expected benchmark results are present."""
        return len(self.missing_concurrencies) == 0


@dataclass
class BatchMetrics:
    """Metrics from a single batch (prefill or decode), parsed from log files."""

    timestamp: str
    dp: int
    tp: int
    ep: int
    batch_type: str  # "prefill" or "decode"
    # Optional metrics
    new_seq: int | None = None
    new_token: int | None = None
    cached_token: int | None = None
    token_usage: float | None = None
    running_req: int | None = None
    queue_req: int | None = None
    prealloc_req: int | None = None
    inflight_req: int | None = None
    input_throughput: float | None = None
    gen_throughput: float | None = None
    transfer_req: int | None = None
    num_tokens: int | None = None
    preallocated_usage: float | None = None

    @property
    def cache_hit_rate(self) -> float | None:
        """Calculate cache hit rate percentage."""
        if self.new_token is not None and self.cached_token is not None:
            total = self.new_token + self.cached_token
            return (self.cached_token / total * 100) if total > 0 else None
        return None


@dataclass
class MemoryMetrics:
    """Memory metrics from log lines."""
    metric_type: str  # "memory" or "kv_cache"
    timestamp: str = "" # trtllm does not echo timestamp in the log lines
    dp: int = 0  # trtllm does not echo dp, tp, ep in the log lines
    tp: int = 0  # trtllm does not echo dp, tp, ep in the log lines
    ep: int = 0  # trtllm does not echo dp, tp, ep in the log lines
    avail_mem_gb: float | None = None
    mem_usage_gb: float | None = None
    kv_cache_gb: float | None = None
    kv_tokens: int | None = None

@dataclass  
class ParsedCommandInfo:
    """Parsed command information from .err/.out log files.

    Attributes:
        explicit_flags: Set of CLI flag names that were explicitly set (e.g., {'tp-size', 'model-path'})
        services: Mapping of node names to their service types (e.g., {'cn01': ['prefill', 'decode']})
        backend_type: Detected backend type ('sglang', 'trtllm', 'dynamo', or None)
        commands: Mapping of node_service keys to full command lines (e.g., {'cn01_prefill': 'python3 -m ...'})
    """

    explicit_flags: set[str] = field(default_factory=set)
    services: dict[str, list[str]] = field(default_factory=dict)
    backend_type: str | None = None
    commands: dict[str, str] = field(default_factory=dict)


# Config-related TypedDicts (from config_reader.py)
class GPUInfo(TypedDict, total=False):
    """Expected structure of GPU info in node config."""

    count: int
    gpus: list[dict[str, Any]]
    name: str
    memory_total: str
    driver_version: str


class ServerArgs(TypedDict, total=False):
    """Server launch arguments from node config.

    Contains parallelism settings, model configuration, and parsed command info.
    Note: This is partial - actual configs may have many more fields.
    """

    tp_size: int  # Tensor parallelism size
    dp_size: int  # Data parallelism size
    pp_size: int  # Pipeline parallelism size
    ep_size: int  # Expert parallelism size (for MoE models)
    served_model_name: str  # Model name exposed via API
    attention_backend: str  # Attention implementation (e.g., 'flashinfer')
    kv_cache_dtype: str  # KV cache data type (e.g., 'fp8_e5m2')
    max_total_tokens: int  # Maximum tokens in KV cache
    chunked_prefill_size: int  # Chunk size for prefill
    disaggregation_mode: str  # Disaggregation mode (e.g., 'prefill', 'decode')
    context_length: int  # Maximum context length
    command: ParsedCommandInfo  # Parsed command line info


@dataclass
class NodeConfig:
    """Node configuration from *_config.json files.

    Attributes:
        filename: Name of the config file (e.g., 'cn01_prefill_w0_config.json')
        gpu_info: GPU hardware information including count, names, and memory
        server_args: Server launch arguments (TP/DP/EP sizes, model config, etc.)
        environment: Environment variables set for this node
    """

    filename: str
    gpu_info: GPUInfo
    server_args: ServerArgs
    environment: dict[str, str] = field(default_factory=dict)

@dataclass
class NodeInfo:
    """Node information from log files."""

    node_name: str
    worker_type: str
    worker_id: str

@dataclass
class NodeMetrics():
    """Metrics from a single node (prefill or decode worker), parsed from log files."""

    node_info: NodeInfo | None = None  # Has node name, worker type, worker_id
    batches: list[BatchMetrics] = field(default_factory=list)
    memory_snapshots: list[MemoryMetrics] = field(default_factory=list)
    config: NodeConfig | None = None  # Full node config from *_config.json
    run_id: str = ""

    @property
    def node_name(self) -> str:
        """Get node name."""
        if hasattr(self.node_info, "node_name"):
            # NodeInfo dataclass
            return self.node_info.node_name
        # Legacy dict format
        return self.node_info.get("node", "Unknown") if self.node_info else "Unknown"

    @property
    def worker_type(self) -> str:
        """Get worker type (prefill/decode/frontend)."""
        if hasattr(self.node_info, "worker_type"):
            # NodeInfo dataclass
            return self.node_info.worker_type
        # Legacy dict format
        return self.node_info.get("worker_type", "unknown") if self.node_info else "unknown"

    @property
    def is_prefill(self) -> bool:
        """Check if this is a prefill node."""
        return self.worker_type == "prefill"

    @property
    def is_decode(self) -> bool:
        """Check if this is a decode node."""
        return self.worker_type == "decode"
"""
Node analysis service for parsing .err/.out log files

All parsing logic encapsulated in the NodeAnalyzer class.
"""

import logging
import os
from pathlib import Path
import re
import yaml
import pandas as pd
from typing import Any

from analysis.srtlog.models import NodeConfig, NodeInfo, NodeMetrics, BatchMetrics, MemoryMetrics
from srtctl.backends import BackendType

from .cache_manager import CacheManager

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NodeAnalyzer:
    """Service for analyzing node-level metrics from log files.

    Parses .err/.out files to extract batch metrics, memory usage, and configuration.
    All parsing logic is encapsulated as methods.
    """

    def parse_run_logs(self, run_path: str, return_dicts: bool = False) -> list[NodeMetrics] | list[dict[str, Any]]:
        """Parse all node log files in a run directory.

        Uses parquet caching to avoid re-parsing on subsequent loads.

        Args:
            run_path: Path to the run directory containing .err/.out files
            return_dicts: If True, return dicts directly (faster). If False, return NodeMetrics objects.

        Returns:
            List of NodeMetrics objects or dicts, one per node
        """
        # Initialize cache manager
        cache_mgr = CacheManager(run_path)

        # Define source patterns for cache validation (.err and .out files)
        source_patterns = ["*.err", "*.out"]

        # Try to load from cache first
        if cache_mgr.is_cache_valid("node_metrics", source_patterns):
            cached_df = cache_mgr.load_from_cache("node_metrics")
            if cached_df is not None and not cached_df.empty:
                if return_dicts:
                    # Fast path: convert directly to dicts without NodeMetrics objects
                    nodes = self._dataframe_to_dicts(cached_df)
                    logger.info(f"Loaded {len(nodes)} nodes from cache (as dicts)")
                else:
                    # Reconstruct NodeMetrics objects from DataFrame
                    nodes = self._deserialize_node_metrics(cached_df)
                    logger.info(f"Loaded {len(nodes)} nodes from cache")
                return nodes

        # Cache miss or invalid - parse from .err/.out files
        nodes = []

        if not os.path.exists(run_path):
            logger.error(f"Run path does not exist: {run_path}")
            return nodes

        # Detect backend type from config.yaml
        backend_type: BackendType | None = self._detect_backend_type_from_config(run_path)
        if backend_type is None:
            logger.error(f"Could not detect backend type from config.yaml in {run_path}")
            return nodes
        
        logger.info(f"Detected backend type: {backend_type}")

        total_err_files = 0
        parsed_successfully = 0

        # Search in both run_path and run_path/logs/ (TRT-LLM uses logs/ subdirectory)
        search_dirs = [run_path]
        logs_subdir = os.path.join(run_path, "logs")
        if os.path.exists(logs_subdir):
            search_dirs.append(logs_subdir)

        for search_dir in search_dirs:
            for file in os.listdir(search_dir):
                if (file.endswith(".err") or file.endswith(".out")) and ("prefill" in file or "decode" in file):
                    total_err_files += 1
                    filepath = os.path.join(search_dir, file)

                    node_info = self._extract_node_info_from_filename(filepath)
                    if not node_info:
                        logger.warning(
                            f"Could not extract node info from filename: {filepath}. "
                            f"Expected format: <node>_<service>_<id>.err or .out"
                        )
                        continue

                    node_config = self._parse_node_config(
                        filepath, backend_type, node_info)
                        
                    batch_metrics = self._parse_batch_metrics(
                        filepath, backend_type, node_info)

                    memory_metrics = self._parse_memory_metrics(
                        filepath, backend_type, node_info)
                    
                    node_metrics = NodeMetrics(
                        node_info=node_info,
                        batches=batch_metrics,
                        memory_snapshots=memory_metrics,
                        config=node_config,
                    )
                    nodes.append(node_metrics)
                    parsed_successfully += 1

        logger.info(
            f"Parsed {parsed_successfully}/{total_err_files} prefill/decode log files from {run_path} "
            f"(backend: {backend_type})"
        )

        if total_err_files == 0:
            logger.warning(f"No prefill/decode log files found in {run_path} or {run_path}/logs/")

        # Save to cache if we have data
        if nodes:
            cache_df = self._serialize_node_metrics(nodes)
            cache_mgr.save_to_cache("node_metrics", cache_df, source_patterns)

        return nodes

    def _parse_node_config(self, filepath: str, backend_type: BackendType, node_info: NodeInfo) -> NodeConfig | None:
        """Parse node configuration from config files located in the log directory

        The log directory is the directory containing the log files.
        trtllm_config_{worker_type}.yaml is the config file for the worker type.
        config.yaml is the main config file. The environment variables are in the main config file.
        
        Args:
        
            filepath: Path to the log file (used to locate config files)
            backend_type: Backend type
            node_info: Node info
        Returns:
            NodeConfig object or None if not found

        Raises:
            NotImplementedError: For SGLang backend (uses different config format)
        """

        if backend_type == BackendType.SGLANG.value:
            raise NotImplementedError("SGLang config parsing not implemented.")
        
        logger.info(f"Parsing node config for {node_info.worker_type} worker type with backend type: {backend_type}")

        # TRT-LLM: Load config from YAML file based on worker type
        log_dir = Path(filepath).parent
        yaml_path = log_dir / f"trtllm_config_{node_info.worker_type}.yaml"

        if not yaml_path.exists():
            logger.warning(f"TRT-LLM config file not found: {yaml_path}")
            return None

        node_config = NodeConfig(
            filename=yaml_path.name,
            gpu_info={},
            server_args={},
            environment={},
        )

        if not yaml_path.exists():
            logger.debug(f"TRT-LLM config file not found: {yaml_path}")
            return None

        # get server config from yaml file

        try:
            with yaml_path.open() as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    # Flatten all nested keys with dot notation
                    node_config.server_args = self._flatten_dict(yaml_config)
        except Exception as e:
            logger.warning(f"Failed to parse {yaml_path}: {e}")

        # get environment from main config.yaml 
        config_path = log_dir / "config.yaml"
        if not config_path.exists():
            config_path = log_dir.parent / "config.yaml"
        with config_path.open() as f:
            config = yaml.safe_load(f)
            if config:
                node_config.environment = config.get("backend").get(f"{node_info.worker_type}_environment", {})

        # Extract command from log file (pattern: "Rank0 run python3 ... in background")
        command = self._extract_command_from_log(filepath)
        if command:
            node_config.server_args["command"] = command

        return node_config

    def _extract_command_from_log(self, filepath: str) -> str | None:
        """Extract the command line from a TRT-LLM log file.

        Looks for lines like: "Rank0 run python3 -m dynamo.trtllm ... in background"

        Args:
            filepath: Path to the log file

        Returns:
            Command string or None if not found
        """
        import re

        command_pattern = re.compile(r"Rank0 run (python3.*?) in background")

        try:
            with open(filepath, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if "Rank0 run python3" in line:
                        # Strip ANSI escape codes
                        clean_line = re.sub(r"\x1b\[[0-9;]*m", "", line)
                        match = command_pattern.search(clean_line)
                        if match:
                            return match.group(1).strip()
        except Exception as e:
            logger.warning(f"Error extracting command from {filepath}: {e}")

        return None

    def _parse_batch_metrics(self, filepath: str, backend_type: BackendType, node_info: NodeInfo) -> list:
        """Parse batch metrics from log file.

        Args:
            filepath: Path to the log file
            backend_type: Backend type
            node_info: Node info

        Returns:
            List of BatchMetrics objects

        Raises:
            NotImplementedError: For SGLang backend
        """
        if backend_type == BackendType.SGLANG:
            raise NotImplementedError(
                "SGLang batch metrics parsing not implemented. "
                "Use parse_single_log() for SGLang logs."
            )

        # TRT-LLM doesn't log batch metrics like SGLang
        # Return empty list for TRT-LLM
        return []

    def _parse_memory_metrics(self, filepath: str, backend_type: BackendType, node_info: NodeInfo) -> list:
        """Parse memory metrics from log file.

        Args:
            filepath: Path to the log file
            backend_type: Backend type
            node_info: Node info

        Returns:
            List of MemoryMetrics objects

        Raises:
            NotImplementedError: For SGLang backend
        """
        if backend_type == BackendType.SGLANG:
            raise NotImplementedError(
                "SGLang memory metrics parsing not implemented. "
                "Use parse_single_log() for SGLang logs."
            )

        # TRT-LLM: Parse KV cache metrics from log
        memory_snapshots = []

        try:
            with open(filepath, encoding="utf-8", errors="replace") as f:
                for line in f:
                    mem_metrics = self._parse_trtllm_memory_line(line)
                    if mem_metrics and mem_metrics.get("type") == "kv_cache":
                        memory_snapshots.append(
                            MemoryMetrics(
                                timestamp=mem_metrics.get("timestamp", ""),
                                metric_type="kv_cache",
                                kv_cache_gb=mem_metrics.get("kv_cache_gb"),
                                kv_tokens=mem_metrics.get("kv_tokens"),
                            )
                        )
        except Exception as e:
            logger.error(f"Error parsing memory metrics from {filepath}: {e}")

        return memory_snapshots

    def get_prefill_nodes(self, nodes: list):
        """Filter for prefill nodes only.

        Args:
            nodes: List of NodeMetrics objects

        Returns:
            Filtered list containing only prefill nodes
        """
        return [n for n in nodes if n.is_prefill]

    def get_decode_nodes(self, nodes: list):
        """Filter for decode nodes only.

        Args:
            nodes: List of NodeMetrics objects

        Returns:
            Filtered list containing only decode nodes
        """
        return [n for n in nodes if n.is_decode]

    def get_node_count(self, run_path: str) -> tuple[int, int]:
        """Get count of prefill and decode nodes in a run.

        Args:
            run_path: Path to the run directory

        Returns:
            Tuple of (prefill_count, decode_count)
        """
        nodes = self.parse_run_logs(run_path)

        prefill_count = sum(1 for n in nodes if n.is_prefill)
        decode_count = sum(1 for n in nodes if n.is_decode)

        return (prefill_count, decode_count)

    def has_batch_metrics(self, nodes: list) -> bool:
        """Check if any node has batch-level metrics.

        Useful for detecting if decode nodes are logging batch metrics.

        Args:
            nodes: List of NodeMetrics objects

        Returns:
            True if any node has batch data
        """
        return any(len(n.batches) > 0 for n in nodes)

    def _serialize_node_metrics(self, nodes: list) -> pd.DataFrame:
        """Serialize NodeMetrics objects to a DataFrame for caching.

        Args:
            nodes: List of NodeMetrics objects

        Returns:
            DataFrame with all batch and memory metrics
        """
        rows = []

        for node in nodes:
            node_info = node.node_info
            config = node.config or {}  # Handle None config

            # Extract node info values (handle both NodeInfo dataclass and dict)
            if hasattr(node_info, "node_name"):
                # NodeInfo dataclass
                node_name = node_info.node_name
                worker_type = node_info.worker_type
                worker_id = node_info.worker_id
            else:
                # Legacy dict format
                node_name = node_info.get("node", "") if node_info else ""
                worker_type = node_info.get("worker_type", "") if node_info else ""
                worker_id = node_info.get("worker_id", "") if node_info else ""

            # Extract config values (handle both dict and NodeConfig types)
            if hasattr(config, "server_args"):
                # NodeConfig dataclass
                server_args = config.server_args or {}
                tp_size = server_args.get("tp_size")
                dp_size = server_args.get("dp_size")
                ep_size = server_args.get("ep_size")
            else:
                # Legacy dict format
                tp_size = config.get("tp_size") if config else None
                dp_size = config.get("dp_size") if config else None
                ep_size = config.get("ep_size") if config else None

            # Serialize batch metrics
            for batch in node.batches:
                row = {
                    # Node identification
                    "node": node_name,
                    "worker_type": worker_type,
                    "worker_id": worker_id,
                    # Config
                    "tp_size": tp_size,
                    "dp_size": dp_size,
                    "ep_size": ep_size,
                    # Metric type
                    "metric_type": "batch",
                    # Batch data
                    "timestamp": batch.timestamp,
                    "dp": batch.dp,
                    "tp": batch.tp,
                    "ep": batch.ep,
                    "batch_type": batch.batch_type,
                    "new_seq": batch.new_seq,
                    "new_token": batch.new_token,
                    "cached_token": batch.cached_token,
                    "token_usage": batch.token_usage,
                    "running_req": batch.running_req,
                    "queue_req": batch.queue_req,
                    "prealloc_req": batch.prealloc_req,
                    "inflight_req": batch.inflight_req,
                    "transfer_req": batch.transfer_req,
                    "preallocated_usage": batch.preallocated_usage,
                    "num_tokens": batch.num_tokens,
                    "input_throughput": batch.input_throughput,
                    "gen_throughput": batch.gen_throughput,
                }
                rows.append(row)

            # Serialize memory metrics
            for mem in node.memory_snapshots:
                row = {
                    # Node identification
                    "node": node_name,
                    "worker_type": worker_type,
                    "worker_id": worker_id,
                    # Config
                    "tp_size": tp_size,
                    "dp_size": dp_size,
                    "ep_size": ep_size,
                    # Metric type
                    "metric_type": "memory",
                    # Memory data
                    "timestamp": mem.timestamp,
                    "dp": mem.dp,
                    "tp": mem.tp,
                    "ep": mem.ep,
                    "avail_mem_gb": mem.avail_mem_gb,
                    "mem_usage_gb": mem.mem_usage_gb,
                    "kv_cache_gb": mem.kv_cache_gb,
                    "kv_tokens": mem.kv_tokens,
                }
                rows.append(row)

        return pd.DataFrame(rows)

    def _deserialize_node_metrics(self, df: pd.DataFrame) -> list:
        """Deserialize NodeMetrics objects from a cached DataFrame.

        Args:
            df: DataFrame with cached node metrics

        Returns:
            List of NodeMetrics objects
        """
        import time
        from .models import BatchMetrics, MemoryMetrics, NodeMetrics

        start_time = time.time()
        nodes = []

        # Group by node
        for (node_name, worker_type, worker_id), group_df in df.groupby(
            ["node", "worker_type", "worker_id"], dropna=False
        ):
            node_info = {
                "node": node_name,
                "worker_type": worker_type,
                "worker_id": worker_id,
            }

            # Extract config (same for all rows in this node)
            config = {}
            if not group_df.empty:
                first_row = group_df.iloc[0]
                if pd.notna(first_row.get("tp_size")):
                    config["tp_size"] = int(first_row["tp_size"])
                if pd.notna(first_row.get("dp_size")):
                    config["dp_size"] = int(first_row["dp_size"])
                if pd.notna(first_row.get("ep_size")):
                    config["ep_size"] = int(first_row["ep_size"])

            # Separate batch and memory metrics
            batch_df = group_df[group_df["metric_type"] == "batch"]
            memory_df = group_df[group_df["metric_type"] == "memory"]

            # Reconstruct batch metrics using vectorized operations
            batches = []
            if not batch_df.empty:
                # Convert to dict records in bulk (much faster than iterrows)
                batch_records = batch_df.to_dict("records")
                for row in batch_records:
                    batch = BatchMetrics(
                        timestamp=row["timestamp"],
                        dp=int(row["dp"]) if pd.notna(row["dp"]) else 0,
                        tp=int(row["tp"]) if pd.notna(row["tp"]) else 0,
                        ep=int(row["ep"]) if pd.notna(row["ep"]) else 0,
                        batch_type=row["batch_type"],
                        new_seq=int(row["new_seq"]) if pd.notna(row.get("new_seq")) else None,
                        new_token=int(row["new_token"]) if pd.notna(row.get("new_token")) else None,
                        cached_token=(int(row["cached_token"]) if pd.notna(row.get("cached_token")) else None),
                        token_usage=row.get("token_usage") if pd.notna(row.get("token_usage")) else None,
                        running_req=(int(row["running_req"]) if pd.notna(row.get("running_req")) else None),
                        queue_req=int(row["queue_req"]) if pd.notna(row.get("queue_req")) else None,
                        prealloc_req=(int(row["prealloc_req"]) if pd.notna(row.get("prealloc_req")) else None),
                        inflight_req=(int(row["inflight_req"]) if pd.notna(row.get("inflight_req")) else None),
                        transfer_req=(int(row["transfer_req"]) if pd.notna(row.get("transfer_req")) else None),
                        preallocated_usage=(
                            row.get("preallocated_usage") if pd.notna(row.get("preallocated_usage")) else None
                        ),
                        num_tokens=int(row["num_tokens"]) if pd.notna(row.get("num_tokens")) else None,
                        input_throughput=(
                            row.get("input_throughput") if pd.notna(row.get("input_throughput")) else None
                        ),
                        gen_throughput=(row.get("gen_throughput") if pd.notna(row.get("gen_throughput")) else None),
                    )
                    batches.append(batch)

            # Reconstruct memory metrics using vectorized operations
            memory_snapshots = []
            if not memory_df.empty:
                # Convert to dict records in bulk (much faster than iterrows)
                memory_records = memory_df.to_dict("records")
                for row in memory_records:
                    mem = MemoryMetrics(
                        timestamp=row["timestamp"],
                        dp=int(row["dp"]) if pd.notna(row["dp"]) else 0,
                        tp=int(row["tp"]) if pd.notna(row["tp"]) else 0,
                        ep=int(row["ep"]) if pd.notna(row["ep"]) else 0,
                        metric_type="memory",
                        avail_mem_gb=(row.get("avail_mem_gb") if pd.notna(row.get("avail_mem_gb")) else None),
                        mem_usage_gb=(row.get("mem_usage_gb") if pd.notna(row.get("mem_usage_gb")) else None),
                        kv_cache_gb=(row.get("kv_cache_gb") if pd.notna(row.get("kv_cache_gb")) else None),
                        kv_tokens=int(row["kv_tokens"]) if pd.notna(row.get("kv_tokens")) else None,
                    )
                    memory_snapshots.append(mem)

            # Create NodeMetrics object
            node = NodeMetrics(
                node_info=node_info,
                batches=batches,
                memory_snapshots=memory_snapshots,
                config=config,
            )
            nodes.append(node)

        elapsed = time.time() - start_time
        logger.info(f"Deserialized {len(nodes)} nodes in {elapsed:.2f}s")
        return nodes

    # Private helper methods

    def _parse_dp_tp_ep_tag(self, line: str) -> tuple[int | None, int | None, int | None, str | None]:
        """Extract DP, TP, EP indices and timestamp from log line.

        Supports three formats:
        - Full: [2025-11-04 05:31:43 DP0 TP0 EP0]
        - Simple TP: [2025-11-04 07:05:55 TP0] (defaults DP=0, EP=0)
        - Pipeline: [2025-12-08 14:34:44 PP0] (defaults DP=0, EP=0, TP=PP value)

        Args:
            line: Log line to parse

        Returns:
            (dp, tp, ep, timestamp) or (None, None, None, None) if pattern not found
        """
        # Try full format first: DP0 TP0 EP0
        match = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) DP(\d+) TP(\d+) EP(\d+)\]", line)
        if match:
            timestamp, dp, tp, ep = match.groups()
            return int(dp), int(tp), int(ep), timestamp

        # Try simple format: TP0 only (1P4D style)
        match = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) TP(\d+)\]", line)
        if match:
            timestamp, tp = match.groups()
            return 0, int(tp), 0, timestamp  # Default DP=0, EP=0

        # Try pipeline parallelism format: PP0 (prefill with PP)
        match = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) PP(\d+)\]", line)
        if match:
            timestamp, pp = match.groups()
            return 0, int(pp), 0, timestamp  # Map PP to TP slot, default DP=0, EP=0

        return None, None, None, None

    def _parse_prefill_batch_line(self, line: str) -> dict | None:
        """Parse prefill batch log line for metrics.

        Example line:
        [2025-11-04 05:31:43 DP0 TP0 EP0] Prefill batch, #new-seq: 18, #new-token: 16384,
        #cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 0,
        #prealloc-req: 0, #inflight-req: 0, input throughput (token/s): 0.00,
        """
        dp, tp, ep, timestamp = self._parse_dp_tp_ep_tag(line)
        if dp is None or "Prefill batch" not in line:
            return None

        metrics = {"timestamp": timestamp, "dp": dp, "tp": tp, "ep": ep, "type": "prefill"}

        # Extract metrics using regex
        patterns = {
            "new_seq": r"#new-seq:\s*(\d+)",
            "new_token": r"#new-token:\s*(\d+)",
            "cached_token": r"#cached-token:\s*(\d+)",
            "token_usage": r"token usage:\s*([\d.]+)",
            "running_req": r"#running-req:\s*(\d+)",
            "queue_req": r"#queue-req:\s*(\d+)",
            "prealloc_req": r"#prealloc-req:\s*(\d+)",
            "inflight_req": r"#inflight-req:\s*(\d+)",
            "input_throughput": r"input throughput \(token/s\):\s*([\d.]+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                value = match.group(1)
                metrics[key] = float(value) if "." in value else int(value)

        return metrics

    def _parse_decode_batch_line(self, line: str) -> dict | None:
        """Parse decode batch log line for metrics.

        Example line:
        [2025-11-04 05:32:32 DP31 TP31 EP31] Decode batch, #running-req: 7, #token: 7040,
        token usage: 0.00, pre-allocated usage: 0.00, #prealloc-req: 0, #transfer-req: 0,
        #retracted-req: 0, cuda graph: True, gen throughput (token/s): 6.73, #queue-req: 0,
        """
        dp, tp, ep, timestamp = self._parse_dp_tp_ep_tag(line)
        if dp is None or "Decode batch" not in line:
            return None

        metrics = {"timestamp": timestamp, "dp": dp, "tp": tp, "ep": ep, "type": "decode"}

        # Extract metrics using regex
        patterns = {
            "running_req": r"#running-req:\s*(\d+)",
            "num_tokens": r"#token:\s*(\d+)",
            "token_usage": r"token usage:\s*([\d.]+)",
            "preallocated_usage": r"pre-allocated usage:\s*([\d.]+)",
            "prealloc_req": r"#prealloc-req:\s*(\d+)",
            "transfer_req": r"#transfer-req:\s*(\d+)",
            "queue_req": r"#queue-req:\s*(\d+)",
            "gen_throughput": r"gen throughput \(token/s\):\s*([\d.]+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                value = match.group(1)
                metrics[key] = float(value) if "." in value else int(value)

        return metrics

    def _parse_memory_line(self, line: str) -> dict | None:
        """Parse memory-related log lines.

        Examples:
        [2025-11-04 05:27:13 DP0 TP0 EP0] Load weight end. type=DeepseekV3ForCausalLM,
        dtype=torch.bfloat16, avail mem=75.11 GB, mem usage=107.07 GB.

        [2025-11-04 05:27:13 DP0 TP0 EP0] KV Cache is allocated. #tokens: 524288, KV size: 17.16 GB
        """
        dp, tp, ep, timestamp = self._parse_dp_tp_ep_tag(line)
        if dp is None:
            return None

        metrics = {
            "timestamp": timestamp,
            "dp": dp,
            "tp": tp,
            "ep": ep,
        }

        # Parse available memory
        avail_match = re.search(r"avail mem=([\d.]+)\s*GB", line)
        if avail_match:
            metrics["avail_mem_gb"] = float(avail_match.group(1))
            metrics["type"] = "memory"

        # Parse memory usage
        usage_match = re.search(r"mem usage=([\d.]+)\s*GB", line)
        if usage_match:
            metrics["mem_usage_gb"] = float(usage_match.group(1))
            metrics["type"] = "memory"

        # Parse KV cache size
        kv_match = re.search(r"KV size:\s*([\d.]+)\s*GB", line)
        if kv_match:
            metrics["kv_cache_gb"] = float(kv_match.group(1))
            metrics["type"] = "kv_cache"

        # Parse token count for KV cache
        token_match = re.search(r"#tokens:\s*(\d+)", line)
        if token_match:
            metrics["kv_tokens"] = int(token_match.group(1))

        return metrics if "type" in metrics else None

    def _flatten_dict(self, d: dict, parent_key: str = "", sep: str = ".") -> dict:
        """Flatten a nested dictionary using dot notation for keys.

        Example:
            {"kv_cache_config": {"dtype": "fp8"}} -> {"kv_cache_config.dtype": "fp8"}

        Args:
            d: Dictionary to flatten
            parent_key: Prefix for keys (used in recursion)
            sep: Separator to use between nested keys

        Returns:
            Flattened dictionary with dot-separated keys
        """
        items: list[tuple[str, any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                # Recurse into nested dicts
                items.extend(self._flatten_dict(v, new_key, sep).items())
            elif isinstance(v, list):
                # Convert lists to comma-separated strings for CSV compatibility
                items.append((new_key, ",".join(str(x) for x in v) if v else None))
            else:
                items.append((new_key, v))
        return dict(items)

    def _extract_node_info_from_filename(self, filename: str) -> NodeInfo | None:
        """Extract node name and worker info from filename.

        Example: watchtower-navy-cn01_prefill_w0.err or r02-p01-dgx-c11_prefill_w0.out
        Returns: NodeInfo(node_name='cn01', worker_type='prefill', worker_id='w0')
        """
        # Use greedy match for node name up to _(prefill|decode|frontend)_
        match = re.match(r"(.+)_(prefill|decode|frontend)_([^.]+)\.(err|out)", os.path.basename(filename))
        if not match:
            return None
        return NodeInfo(
            node_name=match.group(1).replace("watchtower-navy-", ""),
            worker_type=match.group(2),
            worker_id=match.group(3),
        )

    # TRT-LLM specific parsing methods

    def _parse_trtllm_timestamp(self, line: str) -> str | None:
        """Parse TRT-LLM timestamp from log line.

        Supports formats:
        - [01/15/2026-06:43:26] [TRT-LLM] ...
        - [2026-01-15 06:43:25] INFO ...

        Returns:
            Timestamp string in ISO 8601 format, or None
        """
        from datetime import datetime

        # Format: [MM/DD/YYYY-HH:MM:SS]
        match = re.search(r"\[(\d{2})/(\d{2})/(\d{4})-(\d{2}:\d{2}:\d{2})\]", line)
        if match:
            month, day, year, time = match.groups()
            dt = datetime.strptime(f"{year}-{month}-{day} {time}", "%Y-%m-%d %H:%M:%S")
            return dt.isoformat()

        # Format: [YYYY-MM-DD HH:MM:SS]
        match = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]", line)
        if match:
            dt = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
            return dt.isoformat()

        return None

    def _parse_trtllm_config(self, line: str) -> dict | None:
        """Parse TRT-LLM Config(...) line to extract configuration.

        Example:
        Initializing the worker with config: Config(namespace=dynamo, component=prefill,
        tensor_parallel_size=1, pipeline_parallel_size=1, max_batch_size=2048, ...)

        Returns:
            Dict with parsed config values, or None
        """
        if "Config(" not in line:
            return None

        config = {}

        # Extract key=value pairs from Config(...)
        patterns = {
            "component": r"component=(\w+)",
            "tensor_parallel_size": r"tensor_parallel_size=(\d+)",
            "pipeline_parallel_size": r"pipeline_parallel_size=(\d+)",
            "expert_parallel_size": r"expert_parallel_size=(\d+)",
            "max_batch_size": r"max_batch_size=(\d+)",
            "max_num_tokens": r"max_num_tokens=(\d+)",
            "disaggregation_mode": r"disaggregation_mode=DisaggregationMode\.(\w+)",
            "served_model_name": r"served_model_name=([^,\)]+)",
            "model_path": r"model_path=([^,\)]+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                value = match.group(1)
                # Convert numeric strings to int
                if value.isdigit():
                    config[key] = int(value)
                else:
                    config[key] = value

        return config if config else None

    def _parse_trtllm_engine_args(self, line: str) -> dict | None:
        """Parse TRT-LLM engine args line.

        Example:
        TensorRT-LLM engine args: {'tensor_parallel_size': 8, 'pipeline_parallel_size': 1, ...}

        Returns:
            Dict with engine args, or None
        """
        if "TensorRT-LLM engine args:" not in line:
            return None

        config = {}

        patterns = {
            "tp_size": r"'tensor_parallel_size':\s*(\d+)",
            "pp_size": r"'pipeline_parallel_size':\s*(\d+)",
            "ep_size": r"'moe_expert_parallel_size':\s*(\d+)",
            "max_num_tokens": r"'max_num_tokens':\s*(\d+)",
            "backend": r"'backend':\s*'(\w+)'",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                value = match.group(1)
                config[key] = int(value) if value.isdigit() else value

        return config if config else None

    def _parse_trtllm_memory_line(self, line: str) -> dict | None:
        """Parse TRT-LLM memory allocation lines.

        Examples:
        [MemUsageChange] Allocated 0.54 GiB for max tokens in paged KV cache (16576).
        Max KV cache blocks per sequence: 129 [window size=8232], tokens per block=64

        Returns:
            Dict with memory metrics, or None
        """
        timestamp = self._parse_trtllm_timestamp(line)

        metrics = {}

        # Parse memory allocation
        mem_match = re.search(r"\[MemUsageChange\] Allocated ([\d.]+) GiB for max tokens in paged KV cache \((\d+)\)", line)
        if mem_match:
            metrics["kv_cache_gb"] = float(mem_match.group(1))
            metrics["kv_tokens"] = int(mem_match.group(2))
            metrics["type"] = "kv_cache"
            metrics["timestamp"] = timestamp or ""
            return metrics

        # Parse KV cache blocks info
        kv_match = re.search(r"Max KV cache blocks per sequence:\s*(\d+).*tokens per block=(\d+)", line)
        if kv_match:
            metrics["kv_blocks_per_seq"] = int(kv_match.group(1))
            metrics["tokens_per_block"] = int(kv_match.group(2))
            metrics["type"] = "kv_config"
            metrics["timestamp"] = timestamp or ""
            return metrics

        return None

    def _detect_backend_type_from_config(self, run_path: str) -> BackendType:
        """Detect backend type from config.yaml in run directory.

        Looks for backend.type field in config.yaml.

        Args:
            run_path: Path to run directory (or logs subdirectory)

        Returns:
            BackendType
        """
        import yaml

        # Try run_path directly, then parent (in case run_path is logs/)
        paths_to_try = [
            os.path.join(run_path, "config.yaml"),
            os.path.join(os.path.dirname(run_path), "config.yaml"),
        ]

        for config_path in paths_to_try:
            if os.path.exists(config_path):
                try:
                    with open(config_path) as f:
                        config = yaml.safe_load(f)
                    backend_type = BackendType(config.get("backend", {}).get("type", "").lower())
                    logger.info(f"Detected backend type '{backend_type}' from {config_path}")
                    return backend_type
                except Exception as e:
                    logger.warning(f"Error reading config.yaml: {e}")
        return None

# Standalone helper function for visualizations
def get_node_label(node_data: dict) -> str:
    """Generate a display label for a node with its configuration.

    Example: "3320 | 6P1D | 24/32 | cn01-p-w0"
    """
    node_info = node_data["node_info"]
    run_metadata = node_data.get("run_metadata", {})

    # Clean node name
    node_name = (
        node_info["node"].replace("watchtower-navy-", "").replace("watchtower-aqua-", "").replace("inkwell-copper-", "")
    )
    worker_type = node_info["worker_type"][0].lower()  # 'p' for prefill, 'd' for decode
    worker_id = node_info["worker_id"]
    node_short = f"{node_name}-{worker_type}-w{worker_id}"

    # If we have run metadata, use it for context
    if run_metadata:
        job_id = run_metadata.get("job_id", "")
        is_aggregated = run_metadata.get("is_aggregated", False)
        gpus_per_node = run_metadata.get("gpus_per_node", 0)

        if is_aggregated:
            agg_workers = run_metadata.get("agg_workers", 0)
            agg_nodes = run_metadata.get("agg_nodes", 0)
            total_gpus = agg_nodes * gpus_per_node
            # Format: id | xA | total_gpus | node
            return f"{job_id} | {agg_workers}A | {total_gpus} GPUs | {node_short}"
        else:
            prefill_workers = run_metadata.get("prefill_workers", 0)
            decode_workers = run_metadata.get("decode_workers", 0)
            prefill_nodes = run_metadata.get("prefill_nodes", 0)
            decode_nodes = run_metadata.get("decode_nodes", 0)

            prefill_gpus = prefill_nodes * gpus_per_node
            decode_gpus = decode_nodes * gpus_per_node

            # Format: id | xPyD | prefill_gpus/decode_gpus | node
            return f"{job_id} | {prefill_workers}P{decode_workers}D | {prefill_gpus}/{decode_gpus} | {node_short}"
    else:
        # Fallback for old code without metadata
        return node_short

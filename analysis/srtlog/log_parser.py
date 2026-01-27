"""
Node analysis service for parsing .err/.out log files

All parsing logic encapsulated in the NodeAnalyzer class.
"""

import json
import logging
import time
from pathlib import Path

import pandas as pd
import yaml

from .cache_manager import CacheManager
from .models import NodeInfo
from .parsers import get_node_parser

# Configure logging
logger = logging.getLogger(__name__)


class NodeAnalyzer:
    """Service for analyzing node-level metrics from log files.

    Uses the new parser infrastructure to parse node logs based on detected backend type.
    """

    def parse_run_logs(self, run_path: str, return_dicts: bool = False) -> list:
        """Parse all node log files in a run directory.

        Uses parquet caching to avoid re-parsing on subsequent loads.
        Automatically detects backend type and uses appropriate parser.

        Args:
            run_path: Path to the run directory containing .err/.out files
            return_dicts: If True, return dicts directly (faster). If False, return NodeInfo objects.

        Returns:
            List of NodeInfo objects (or dicts), one per node
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
                    # Reconstruct NodeInfo objects from DataFrame
                    nodes = self._deserialize_node_metrics(cached_df, run_path=run_path)
                    logger.info(f"Loaded {len(nodes)} nodes from cache")
                return nodes

        # Cache miss or invalid - parse using new parser infrastructure
        backend_type = self._detect_backend_type(run_path)
        if not backend_type:
            logger.warning(f"Could not detect backend type for {run_path}")
            return []

        # Get appropriate parser
        try:
            parser = get_node_parser(backend_type)
        except ValueError as e:
            logger.warning(f"No parser registered for backend '{backend_type}': {e}")
            return []

        # Use parser to parse logs directory
        logs_dir = Path(run_path) / "logs"
        if not logs_dir.exists():
            # For backwards compatibility, try parsing files in run_path directly
            logs_dir = Path(run_path)

        logger.info(f"Using {backend_type} parser to parse logs in {logs_dir}")
        node_infos = parser.parse_logs(logs_dir)

        # Populate additional config from config files if available
        if node_infos:
            self._populate_config_from_files(run_path, node_infos)

        # Save to cache if we have data
        if node_infos:
            # Extract metrics for caching
            metrics_list = [ni.metrics for ni in node_infos]
            cache_df = self._serialize_node_metrics(metrics_list)
            cache_mgr.save_to_cache("node_metrics", cache_df, source_patterns)
            logger.info(f"Parsed and cached {len(node_infos)} nodes from {logs_dir}")

        if return_dicts:
            return [self._node_info_to_dict(node) for node in node_infos]
        return node_infos

    def _detect_backend_type(self, run_path: str) -> str | None:
        """Detect backend type from run metadata.

        Looks for *.json files with container information in run_path
        and its parent directory (for cases where run_path is logs/).
        Also looks at log file content as fallback.

        Args:
            run_path: Path to the run directory (or logs subdirectory)

        Returns:
            Backend type string (e.g., 'sglang', 'trtllm') or None
        """
        run_path = Path(run_path)

        # Try current directory and parent directory
        search_dirs = [run_path]
        if run_path.name == "logs" and run_path.parent.exists():
            search_dirs.insert(0, run_path.parent)  # Check parent first

        # Try JSON files first
        for search_dir in search_dirs:
            json_files = list(search_dir.glob("*.json"))
            for json_file in json_files:
                try:
                    with open(json_file) as f:
                        metadata = json.load(f)
                        # Try different possible locations for container info
                        container = metadata.get("container", "")
                        if not container:
                            container = metadata.get("model", {}).get("container", "")

                        container_lower = container.lower()
                        if "sglang" in container_lower:
                            logger.debug(f"Detected sglang from {json_file}")
                            return "sglang"
                        if "trtllm" in container_lower or "dynamo" in container_lower:
                            logger.debug(f"Detected trtllm from {json_file}")
                            return "trtllm"
                except Exception as e:
                    logger.debug(f"Could not read {json_file}: {e}")
                    continue

        # Try config.yaml as fallback
        for search_dir in search_dirs:
            yaml_path = search_dir / "config.yaml"
            if yaml_path.exists():
                try:
                    with open(yaml_path) as f:
                        config = yaml.safe_load(f)
                        backend_type = config.get("backend", {}).get("type", "").lower()
                        if backend_type in ["sglang", "trtllm"]:
                            logger.debug(f"Detected {backend_type} from config.yaml")
                            return backend_type
                except Exception as e:
                    logger.debug(f"Could not read {yaml_path}: {e}")

        # Last resort: look at log files
        logs_dir = run_path if run_path.name == "logs" else run_path / "logs"
        if logs_dir.exists():
            log_files = list(logs_dir.glob("*.out")) + list(logs_dir.glob("*.err"))
            for log_file in log_files[:3]:  # Check first few files
                try:
                    with open(log_file) as f:
                        content = f.read(2000)  # Read first 2KB
                        if "sglang.launch_server" in content or "sglang.srt" in content:
                            logger.debug(f"Detected sglang from log content in {log_file.name}")
                            return "sglang"
                        if "dynamo.trtllm" in content or "tensorrt_llm" in content:
                            logger.debug(f"Detected trtllm from log content in {log_file.name}")
                            return "trtllm"
                except Exception as e:
                    logger.debug(f"Could not read {log_file}: {e}")

        return None

    def _populate_config_from_files(self, run_path: str, node_infos: list) -> None:
        """Populate node configuration from config files.

        Reads both:
        1. Per-node *_config.json files (gpu_info, server_args)
        2. Global config.yaml file (environment variables by worker type)

        Merges with existing config that already has launch_command from log parsing.

        Args:
            run_path: Path to the run directory (or logs subdirectory)
            node_infos: List of NodeInfo objects to enhance with config file data
        """
        import os

        run_path = Path(run_path)

        # If run_path is the logs directory, look in parent for config files
        if run_path.name == "logs" and run_path.parent.exists():
            config_dir = run_path.parent
        else:
            config_dir = run_path

        # Parse global config.yaml for environment variables
        yaml_env = self._parse_yaml_environment(config_dir)

        # Find all per-node config files
        config_files = {}
        for file in os.listdir(config_dir):
            if file.endswith("_config.json"):
                # Extract node identifier from filename (e.g., "worker-3_prefill_w0_config.json" -> "worker-3_prefill_w0")
                node_id = file.replace("_config.json", "")
                config_files[node_id] = config_dir / file

        # Build or enhance node_config for each NodeInfo
        for node_info in node_infos:
            metrics = node_info.metrics
            node_name = metrics.node_name
            worker_type = metrics.worker_type
            worker_id = metrics.worker_id

            # Try to find matching config file
            # Format: <node>_<worker_type>_<worker_id>_config.json
            potential_keys = [
                f"{node_name}_{worker_type}_{worker_id}",  # Exact match
                f"{node_name}_{worker_type}",  # Without worker_id
                node_name,  # Just node name
            ]

            config_path = None
            for key in potential_keys:
                if key in config_files:
                    config_path = config_files[key]
                    break

            # Load config file if it exists and merge with existing config
            if config_path and config_path.exists():
                try:
                    with open(config_path) as f:
                        file_config = json.load(f)
                        # Merge file config with existing config (which has launch_command)
                        if node_info.node_config:
                            # Keep launch_command from log parsing
                            launch_cmd = node_info.node_config.get("launch_command")
                            node_info.node_config.update(file_config)
                            if launch_cmd:
                                node_info.node_config["launch_command"] = launch_cmd
                        else:
                            node_info.node_config = file_config
                        logger.debug(
                            f"Loaded config for {node_name} with {len(file_config.get('environment', {}))} env vars"
                        )
                except Exception as e:
                    logger.warning(f"Could not load config from {config_path}: {e}")
                    # Keep existing minimal config with launch_command
            else:
                # No config file found
                if not node_info.node_config:
                    node_info.node_config = {"environment": {}}
                elif "environment" not in node_info.node_config:
                    node_info.node_config["environment"] = {}
                logger.debug(f"No config file found for node {node_name}, using minimal config")

            # Merge environment variables from config.yaml
            if yaml_env and worker_type in yaml_env:
                if not node_info.node_config:
                    node_info.node_config = {}
                if "environment" not in node_info.node_config:
                    node_info.node_config["environment"] = {}

                # Merge YAML env vars (they take precedence over JSON)
                yaml_worker_env = yaml_env[worker_type]
                node_info.node_config["environment"].update(yaml_worker_env)
                logger.debug(f"Merged {len(yaml_worker_env)} env vars from config.yaml for {node_name} ({worker_type})")

    def _parse_yaml_environment(self, run_path: Path) -> dict[str, dict[str, str]]:
        """Parse environment variables from config.yaml.

        Args:
            run_path: Path to the run directory

        Returns:
            Dict mapping worker_type to environment variables
            Example: {"prefill": {"VAR1": "val1"}, "decode": {"VAR2": "val2"}}
        """
        yaml_path = run_path / "config.yaml"
        if not yaml_path.exists():
            logger.debug(f"No config.yaml found in {run_path}")
            return {}

        try:
            with open(yaml_path) as f:
                config = yaml.safe_load(f)

            if not config or "backend" not in config:
                logger.debug("config.yaml has no backend section")
                return {}

            backend = config["backend"]
            env_vars = {}

            # Extract prefill_environment
            if "prefill_environment" in backend:
                env_vars["prefill"] = backend["prefill_environment"]
                logger.info(f"Loaded {len(env_vars['prefill'])} prefill env vars from config.yaml")

            # Extract decode_environment
            if "decode_environment" in backend:
                env_vars["decode"] = backend["decode_environment"]
                logger.info(f"Loaded {len(env_vars['decode'])} decode env vars from config.yaml")

            # Extract agg_environment if present
            if "agg_environment" in backend:
                env_vars["agg"] = backend["agg_environment"]
                logger.info(f"Loaded {len(env_vars['agg'])} agg env vars from config.yaml")

            return env_vars

        except Exception as e:
            logger.warning(f"Could not parse config.yaml in {run_path}: {e}")
            return {}

    def _node_info_to_dict(self, node_info: "NodeInfo") -> dict:
        """Convert NodeInfo object to dict for compatibility.

        Args:
            node_info: NodeInfo object

        Returns:
            Dict representation compatible with old structure
        """
        metrics = node_info.metrics
        return {
            "node_info": {
                "node": metrics.node_name,
                "worker_type": metrics.worker_type,
                "worker_id": metrics.worker_id,
            },
            "prefill_batches": metrics.batches,  # Keep as list of BatchMetrics objects
            "memory_snapshots": metrics.memory_snapshots,  # Keep as list of MemoryMetrics objects
            "config": metrics.config,  # Runtime config (TP/PP/EP, batch sizes)
            "node_config": node_info.node_config,  # Full config (environment, launch_command, gpu_info)
            "launch_command": node_info.launch_command,  # Property accessor for backward compatibility
            "environment": node_info.environment,  # Property accessor for backward compatibility
            "run_id": metrics.run_id,
        }

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
            metadata = node.metadata
            config = node.config

            # Serialize batch metrics
            for batch in node.batches:
                row = {
                    # Node identification
                    "node": metadata.node_name,
                    "worker_type": metadata.worker_type,
                    "worker_id": metadata.worker_id,
                    # Config
                    "tp_size": config.get("tp_size"),
                    "dp_size": config.get("dp_size"),
                    "ep_size": config.get("ep_size"),
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
                    "node": metadata.node_name,
                    "worker_type": metadata.worker_type,
                    "worker_id": metadata.worker_id,
                    # Config
                    "tp_size": config.get("tp_size"),
                    "dp_size": config.get("dp_size"),
                    "ep_size": config.get("ep_size"),
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

    def _deserialize_node_metrics(self, df: pd.DataFrame, run_path: str = None) -> list:
        """Deserialize NodeInfo objects from a cached DataFrame.

        Args:
            df: DataFrame with cached node metrics
            run_path: Path to the run directory (for loading config files)

        Returns:
            List of NodeInfo objects
        """
        from .models import BatchMetrics, MemoryMetrics, NodeInfo, NodeMetadata, NodeMetrics

        start_time = time.time()
        nodes = []

        # Group by node
        for (node_name, worker_type, worker_id), group_df in df.groupby(
            ["node", "worker_type", "worker_id"], dropna=False
        ):
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

            # Create NodeMetadata
            node_metadata = NodeMetadata(
                node_name=node_name,
                worker_type=worker_type,
                worker_id=worker_id,
            )

            # Create NodeMetrics (NEW structure)
            metrics = NodeMetrics(
                metadata=node_metadata,
                batches=batches,
                memory_snapshots=memory_snapshots,
                config=config,
            )

            # Create NodeInfo with empty config (will be populated below)
            node_info = NodeInfo(metrics=metrics, node_config={})
            nodes.append(node_info)

        elapsed = time.time() - start_time
        logger.info(f"Deserialized {len(nodes)} nodes in {elapsed:.2f}s")

        # Populate config from files (environment, launch_command)
        if run_path and nodes:
            self._populate_config_from_files(run_path, nodes)

        return nodes


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

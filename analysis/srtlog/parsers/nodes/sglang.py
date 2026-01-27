# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang node log parser.
Parses logs with format:
    [2m2025-12-30T15:52:38.206058Z[0m [32m INFO[0m ... Decode batch, #running-req: 5, ...
This parser handles SGLang structured logging format with ISO 8601 timestamps.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from analysis.srtlog.models import BatchMetrics, MemoryMetrics, NodeInfo, NodeMetadata, NodeMetrics
from analysis.srtlog.parsers import register_node_parser

if TYPE_CHECKING:
    from analysis.srtlog.parsers import NodeLaunchCommand

logger = logging.getLogger(__name__)


# ANSI escape code pattern for stripping colors
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


@register_node_parser("sglang")
class SGLangNodeParser:
    """Parser for SGLang node logs.
    Handles SGLang structured logging with ISO 8601 timestamps.
    May contain ANSI color codes which are stripped during parsing.
    
    Timestamp format: YYYY-MM-DDTHH:MM:SS.microsZ (e.g., 2025-12-30T15:52:38.206058Z)
    """

    @property
    def backend_type(self) -> str:
        return "sglang"
    
    @staticmethod
    def parse_timestamp(timestamp: str) -> datetime:
        """Parse SGLang timestamp format to datetime object.
        
        Args:
            timestamp: Timestamp string in ISO 8601 format (e.g., 2025-12-30T15:52:38.206058Z)
            
        Returns:
            datetime object
            
        Raises:
            ValueError: If timestamp format is invalid
        """
        # Handle both with and without microseconds and timezone
        timestamp = timestamp.rstrip('Z')
        if '.' in timestamp:
            return datetime.fromisoformat(timestamp)
        else:
            return datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S")

    def parse_logs(self, log_dir: Path) -> list[NodeInfo]:
        """Parse all prefill/decode/agg log files in a directory.
        Args:
            log_dir: Directory containing *_prefill_*.out, *_decode_*.out, *_agg_*.out files
        Returns:
            List of NodeInfo objects
        """
        log_dir = Path(log_dir)
        nodes = []

        if not log_dir.exists():
            logger.error("Log directory does not exist: %s", log_dir)
            return nodes

        # Find all worker log files
        for file in os.listdir(log_dir):
            if not (file.endswith(".err") or file.endswith(".out")):
                continue
            if not any(wt in file for wt in ("prefill", "decode", "agg")):
                continue

            filepath = log_dir / file
            node = self.parse_single_log(filepath)
            if node:
                nodes.append(node)

        logger.info("Parsed %d node log files from %s", len(nodes), log_dir)
        return nodes

    def parse_single_log(self, log_path: Path) -> NodeInfo | None:
        """Parse a single node log file.
        Args:
            log_path: Path to a prefill/decode/agg log file
        Returns:
            NodeInfo object or None if parsing failed
        """
        node_info = self._extract_node_info_from_filename(str(log_path))
        if not node_info:
            logger.warning(
                "Could not extract node info from filename: %s. "
                "Expected format: <node>_<service>_<id>.err or .out",
                log_path,
            )
            return None

        batches = []
        memory_snapshots = []
        config = {}
        launch_command = None
        full_content = []

        try:
            with open(log_path) as f:
                for line in f:
                    full_content.append(line)
                    # Strip ANSI escape codes
                    clean_line = ANSI_ESCAPE.sub("", line)

                    # Parse prefill batch metrics
                    batch_metrics = self._parse_prefill_batch_line(clean_line)
                    if batch_metrics:
                        batches.append(
                            BatchMetrics(
                                timestamp=batch_metrics["timestamp"],
                                dp=0,  # Default since not in log
                                tp=0,
                                ep=0,
                                batch_type=batch_metrics["type"],
                                new_seq=batch_metrics.get("new_seq"),
                                new_token=batch_metrics.get("new_token"),
                                cached_token=batch_metrics.get("cached_token"),
                                token_usage=batch_metrics.get("token_usage"),
                                running_req=batch_metrics.get("running_req"),
                                queue_req=batch_metrics.get("queue_req"),
                                prealloc_req=batch_metrics.get("prealloc_req"),
                                inflight_req=batch_metrics.get("inflight_req"),
                                input_throughput=batch_metrics.get("input_throughput"),
                            )
                        )

                    # Parse decode batch metrics
                    decode_metrics = self._parse_decode_batch_line(clean_line)
                    if decode_metrics:
                        batches.append(
                            BatchMetrics(
                                timestamp=decode_metrics["timestamp"],
                                dp=0,
                                tp=0,
                                ep=0,
                                batch_type=decode_metrics["type"],
                                running_req=decode_metrics.get("running_req"),
                                queue_req=decode_metrics.get("queue_req"),
                                prealloc_req=decode_metrics.get("prealloc_req"),
                                transfer_req=decode_metrics.get("transfer_req"),
                                token_usage=decode_metrics.get("token_usage"),
                                preallocated_usage=decode_metrics.get("preallocated_usage"),
                                num_tokens=decode_metrics.get("num_tokens"),
                                gen_throughput=decode_metrics.get("gen_throughput"),
                            )
                        )

                    # Parse memory metrics
                    mem_metrics = self._parse_memory_line(clean_line)
                    if mem_metrics:
                        memory_snapshots.append(
                            MemoryMetrics(
                                timestamp=mem_metrics["timestamp"],
                                dp=0,
                                tp=0,
                                ep=0,
                                metric_type=mem_metrics["type"],
                                avail_mem_gb=mem_metrics.get("avail_mem_gb"),
                                mem_usage_gb=mem_metrics.get("mem_usage_gb"),
                                kv_cache_gb=mem_metrics.get("kv_cache_gb"),
                                kv_tokens=mem_metrics.get("kv_tokens"),
                            )
                        )

                    # Extract TP/DP/EP configuration from server_args
                    if "tp_size=" in clean_line:
                        tp_match = re.search(r"tp_size=(\d+)", clean_line)
                        dp_match = re.search(r"dp_size=(\d+)", clean_line)
                        ep_match = re.search(r"ep_size=(\d+)", clean_line)

                        if tp_match:
                            config["tp_size"] = int(tp_match.group(1))
                        if dp_match:
                            config["dp_size"] = int(dp_match.group(1))
                        if ep_match:
                            config["ep_size"] = int(ep_match.group(1))

            # Parse launch command from full content
            launch_command = self.parse_launch_command("".join(full_content), node_info["worker_type"])

        except Exception as e:
            logger.error("Error parsing %s: %s", log_path, e)
            return None

        total_metrics = len(batches) + len(memory_snapshots)
        if total_metrics == 0:
            logger.debug("Parsed %s but found no batch/memory metrics", log_path)

        logger.debug("Parsed %s: %d batches, %d memory snapshots", log_path, len(batches), len(memory_snapshots))

        # Create NodeMetadata
        node_metadata = NodeMetadata(
            node_name=node_info["node"],
            worker_type=node_info["worker_type"],
            worker_id=node_info["worker_id"],
        )
        
        # Create NodeMetrics with metadata
        metrics = NodeMetrics(
            metadata=node_metadata,
            batches=batches,
            memory_snapshots=memory_snapshots,
            config=config,
        )
        
        # Create NodeConfig with launch_command
        node_config = {}
        if launch_command:
            node_config["launch_command"] = launch_command
            node_config["environment"] = {}  # Will be populated by NodeAnalyzer if config file exists
        
        # Return complete NodeInfo
        return NodeInfo(metrics=metrics, node_config=node_config if node_config else None)

    def _parse_timestamp(self, line: str) -> str | None:
        """Extract ISO 8601 timestamp from log line.
        Example: 2025-12-30T15:52:38.206058Z
        Returns the timestamp string as-is without conversion.
        """
        match = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z?)", line)
        if match:
            return match.group(1)
        return None

    def _parse_prefill_batch_line(self, line: str) -> dict | None:
        """Parse prefill batch log line for metrics."""
        if "Prefill batch" not in line:
            return None

        timestamp = self._parse_timestamp(line)
        if not timestamp:
            return None

        metrics = {"timestamp": timestamp, "type": "prefill"}

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
        """Parse decode batch log line for metrics."""
        if "Decode batch" not in line:
            return None

        timestamp = self._parse_timestamp(line)
        if not timestamp:
            return None

        metrics = {"timestamp": timestamp, "type": "decode"}

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
        """Parse memory-related log lines."""
        timestamp = self._parse_timestamp(line)
        if not timestamp:
            return None

        metrics = {"timestamp": timestamp}

        # Parse available memory from "avail mem=75.11 GB"
        avail_match = re.search(r"avail mem=([\d.]+)\s*GB", line)
        if avail_match:
            metrics["avail_mem_gb"] = float(avail_match.group(1))
            metrics["type"] = "memory"

        # Parse memory usage from "mem usage=107.07 GB"
        usage_match = re.search(r"mem usage=([\d.]+)\s*GB", line)
        if usage_match:
            metrics["mem_usage_gb"] = float(usage_match.group(1))
            metrics["type"] = "memory"

        # Parse KV cache size from "KV size: 17.16 GB"
        kv_match = re.search(r"KV size:\s*([\d.]+)\s*GB", line)
        if kv_match:
            metrics["kv_cache_gb"] = float(kv_match.group(1))
            metrics["type"] = "kv_cache"

        # Parse token count from "#tokens: 524288"
        token_match = re.search(r"#tokens:\s*(\d+)", line)
        if token_match:
            metrics["kv_tokens"] = int(token_match.group(1))

        # Parse from "Capturing batches" progress lines
        # Example: "Capturing batches (bs=256 avail_mem=6.32 GB)"
        capture_match = re.search(r"avail_mem=([\d.]+)\s*GB", line)
        if capture_match and "type" not in metrics:
            metrics["avail_mem_gb"] = float(capture_match.group(1))
            metrics["type"] = "memory"

        return metrics if "type" in metrics else None

    def _extract_node_info_from_filename(self, filename: str) -> dict | None:
        """Extract node name and worker info from filename.
        Example: eos0219_prefill_w0.out
        Returns: {'node': 'eos0219', 'worker_type': 'prefill', 'worker_id': 'w0'}
        """
        match = re.match(
            r"(.+)_(prefill|decode|agg|frontend)_([^.]+)\.(err|out)",
            os.path.basename(filename),
        )
        if match:
            return {
                "node": match.group(1),
                "worker_type": match.group(2),
                "worker_id": match.group(3),
            }
        return None

    def parse_launch_command(self, log_content: str, worker_type: str = "unknown") -> NodeLaunchCommand | None:
        """Parse the SGLang worker launch command from log content.
        Looks for command lines or ServerArgs in the log.
        Args:
            log_content: Content of the worker log file
            worker_type: Type of worker (prefill, decode, agg)
        Returns:
            NodeLaunchCommand with parsed parameters, or None if not found
        """
        from analysis.srtlog.parsers import NodeLaunchCommand

        # Strip ANSI codes for cleaner parsing
        clean_content = ANSI_ESCAPE.sub("", log_content)

        raw_command = None

        # First, try to find [CMD] tagged command (preferred - from our scripts)
        cmd_match = re.search(r"\[CMD\]\s*(.+)$", clean_content, re.MULTILINE)
        if cmd_match:
            raw_command = cmd_match.group(1).strip()

        # Fallback: pattern to match sglang launch commands
        if not raw_command:
            patterns = [
                r"(python[3]?\s+-m\s+sglang\.launch_server\s+[^\n]+)",
                r"(python[3]?\s+.*launch_server\.py\s+[^\n]+)",
                r"(sglang\.launch_server\s+[^\n]+)",
            ]

            for pattern in patterns:
                match = re.search(pattern, clean_content, re.IGNORECASE)
                if match:
                    raw_command = match.group(1).strip()
                    break

        # Also try to parse from ServerArgs() log line
        if not raw_command:
            server_args_match = re.search(r"server_args=ServerArgs\((.*?)\)", clean_content, re.DOTALL)
            if server_args_match:
                raw_command = f"ServerArgs({server_args_match.group(1)[:200]}...)"

        if not raw_command:
            return None

        extra_args: dict[str, Any] = {}

        # Parse SGLang server arguments (from command line)
        arg_patterns = {
            "model_path": r"--model(?:-path)?[=\s]+([^\s]+)",
            "served_model_name": r"--served-model-name[=\s]+([^\s]+)",
            "tp_size": r"--tp-size[=\s]+(\d+)",
            "pp_size": r"--pp-size[=\s]+(\d+)",
            "dp_size": r"--dp-size[=\s]+(\d+)",
            "ep_size": r"--ep-size[=\s]+(\d+)",
            "host": r"--host[=\s]+([^\s]+)",
            "port": r"--port[=\s]+(\d+)",
            "max_num_seqs": r"--max-(?:num-seqs|running-requests)[=\s]+(\d+)",
            "max_model_len": r"--(?:max-model-len|context-length)[=\s]+(\d+)",
            "kv_cache_dtype": r"--kv-cache-dtype[=\s]+([^\s]+)",
            "gpu_memory_utilization": r"--(?:mem-fraction-static|gpu-memory-utilization)[=\s]+([\d.]+)",
            "disaggregation_mode": r"--disaggregation-mode[=\s]+([^\s]+)",
            "nccl_init_addr": r"--(?:dist-init-addr|nccl-init-addr)[=\s]+([^\s]+)",
        }

        # Also parse from ServerArgs format
        server_args_patterns = {
            "model_path": r"model_path=['\"]?([^'\"]+)['\"]?",
            "served_model_name": r"served_model_name=['\"]?([^'\"]+)['\"]?",
            "tp_size": r"tp_size=(\d+)",
            "pp_size": r"pp_size=(\d+)",
            "dp_size": r"dp_size=(\d+)",
            "ep_size": r"ep_size=(\d+)",
            "host": r"host=['\"]?([^'\"]+)['\"]?",
            "port": r"port=(\d+)",
            "max_num_seqs": r"max_running_requests=(\d+)",
            "max_model_len": r"context_length=(\d+)",
            "disaggregation_mode": r"disaggregation_mode=['\"]?([^'\"]+)['\"]?",
        }

        for field, pattern in arg_patterns.items():
            match = re.search(pattern, raw_command)
            if match:
                value: Any = match.group(1)
                if field in ("tp_size", "pp_size", "dp_size", "ep_size", "port", "max_num_seqs", "max_model_len"):
                    value = int(value)
                elif field == "gpu_memory_utilization":
                    value = float(value)
                extra_args[field] = value

        # Try ServerArgs patterns for any missing fields
        for field, pattern in server_args_patterns.items():
            if field not in extra_args:
                match = re.search(pattern, clean_content)
                if match:
                    value = match.group(1)
                    if field in ("tp_size", "pp_size", "dp_size", "ep_size", "port", "max_num_seqs", "max_model_len"):
                        value = int(value)
                    extra_args[field] = value

        return NodeLaunchCommand(
            backend_type="sglang",
            worker_type=worker_type,
            raw_command=raw_command,
            extra_args=extra_args,
        )

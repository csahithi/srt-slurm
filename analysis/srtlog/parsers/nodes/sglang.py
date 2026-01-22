# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang node log parser (v1 format with DP/TP/EP tags).

Parses logs with format:
    [2025-11-04 05:31:43 DP0 TP0 EP0] Prefill batch, #new-seq: 18, #new-token: 16384, ...
    [2025-11-04 05:32:32 DP31 TP31 EP31] Decode batch, #running-req: 7, #token: 7040, ...
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from analysis.srtlog.models import BatchMetrics, MemoryMetrics, NodeMetrics
from analysis.srtlog.parsers import register_node_parser

if TYPE_CHECKING:
    from analysis.srtlog.parsers import NodeLaunchCommand

logger = logging.getLogger(__name__)


@register_node_parser("sglang")
class SGLangNodeParser:
    """Parser for SGLang node logs (v1 format with DP/TP/EP tags).

    This format is used by older SGLang versions that include DP/TP/EP
    indices in the log prefix.
    """

    @property
    def backend_type(self) -> str:
        return "sglang"

    def parse_logs(self, log_dir: Path) -> list[NodeMetrics]:
        """Parse all prefill/decode/agg log files in a directory.

        Args:
            log_dir: Directory containing *_prefill_*.out, *_decode_*.out, *_agg_*.out files

        Returns:
            List of NodeMetrics objects
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

    def parse_single_log(self, log_path: Path) -> NodeMetrics | None:
        """Parse a single node log file.

        Args:
            log_path: Path to a prefill/decode/agg log file

        Returns:
            NodeMetrics object or None if parsing failed
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

        try:
            with open(log_path) as f:
                for line in f:
                    # Parse prefill batch metrics
                    batch_metrics = self._parse_prefill_batch_line(line)
                    if batch_metrics:
                        batches.append(
                            BatchMetrics(
                                timestamp=batch_metrics["timestamp"],
                                dp=batch_metrics["dp"],
                                tp=batch_metrics["tp"],
                                ep=batch_metrics["ep"],
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
                    decode_metrics = self._parse_decode_batch_line(line)
                    if decode_metrics:
                        batches.append(
                            BatchMetrics(
                                timestamp=decode_metrics["timestamp"],
                                dp=decode_metrics["dp"],
                                tp=decode_metrics["tp"],
                                ep=decode_metrics["ep"],
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
                    mem_metrics = self._parse_memory_line(line)
                    if mem_metrics:
                        memory_snapshots.append(
                            MemoryMetrics(
                                timestamp=mem_metrics["timestamp"],
                                dp=mem_metrics["dp"],
                                tp=mem_metrics["tp"],
                                ep=mem_metrics["ep"],
                                metric_type=mem_metrics["type"],
                                avail_mem_gb=mem_metrics.get("avail_mem_gb"),
                                mem_usage_gb=mem_metrics.get("mem_usage_gb"),
                                kv_cache_gb=mem_metrics.get("kv_cache_gb"),
                                kv_tokens=mem_metrics.get("kv_tokens"),
                            )
                        )

                    # Extract TP/DP/EP configuration from command line
                    if "--tp-size" in line:
                        tp_match = re.search(r"--tp-size\s+(\d+)", line)
                        dp_match = re.search(r"--dp-size\s+(\d+)", line)
                        ep_match = re.search(r"--ep-size\s+(\d+)", line)

                        if tp_match:
                            config["tp_size"] = int(tp_match.group(1))
                        if dp_match:
                            config["dp_size"] = int(dp_match.group(1))
                        if ep_match:
                            config["ep_size"] = int(ep_match.group(1))

        except Exception as e:
            logger.error("Error parsing %s: %s", log_path, e)
            return None

        total_metrics = len(batches) + len(memory_snapshots)
        if total_metrics == 0:
            logger.warning(
                "Parsed %s but found no metrics. "
                "Expected to find lines with DP/TP/EP tags. "
                "Log format may have changed.",
                log_path,
            )

        logger.debug("Parsed %s: %d batches, %d memory snapshots", log_path, len(batches), len(memory_snapshots))

        return NodeMetrics(
            node_info=node_info,
            batches=batches,
            memory_snapshots=memory_snapshots,
            config=config,
        )

    def _parse_dp_tp_ep_tag(self, line: str) -> tuple[int | None, int | None, int | None, str | None]:
        """Extract DP, TP, EP indices and timestamp from log line.

        Supports three formats:
        - Full: [2025-11-04 05:31:43 DP0 TP0 EP0]
        - Simple TP: [2025-11-04 07:05:55 TP0] (defaults DP=0, EP=0)
        - Pipeline: [2025-12-08 14:34:44 PP0] (defaults DP=0, EP=0, TP=PP value)
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
            return 0, int(tp), 0, timestamp

        # Try pipeline parallelism format: PP0 (prefill with PP)
        match = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) PP(\d+)\]", line)
        if match:
            timestamp, pp = match.groups()
            return 0, int(pp), 0, timestamp

        return None, None, None, None

    def _parse_prefill_batch_line(self, line: str) -> dict | None:
        """Parse prefill batch log line for metrics."""
        dp, tp, ep, timestamp = self._parse_dp_tp_ep_tag(line)
        if dp is None or "Prefill batch" not in line:
            return None

        metrics = {"timestamp": timestamp, "dp": dp, "tp": tp, "ep": ep, "type": "prefill"}

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
        dp, tp, ep, timestamp = self._parse_dp_tp_ep_tag(line)
        if dp is None or "Decode batch" not in line:
            return None

        metrics = {"timestamp": timestamp, "dp": dp, "tp": tp, "ep": ep, "type": "decode"}

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
        dp, tp, ep, timestamp = self._parse_dp_tp_ep_tag(line)
        if dp is None:
            return None

        metrics = {"timestamp": timestamp, "dp": dp, "tp": tp, "ep": ep}

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

    def _extract_node_info_from_filename(self, filename: str) -> dict | None:
        """Extract node name and worker info from filename.

        Example: watchtower-navy-cn01_prefill_w0.err or r02-p01-dgx-c11_prefill_w0.out
        Returns: {'node': 'watchtower-navy-cn01', 'worker_type': 'prefill', 'worker_id': 'w0'}
        """
        # Use greedy match for node name up to _(prefill|decode|agg|frontend)_
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

        Looks for command lines like:
            python -m sglang.launch_server --model ... --tp-size ...

        Args:
            log_content: Content of the worker log file
            worker_type: Type of worker (prefill, decode, agg)

        Returns:
            NodeLaunchCommand with parsed parameters, or None if not found
        """
        from analysis.srtlog.parsers import NodeLaunchCommand

        # Pattern to match sglang launch commands
        patterns = [
            r"(python[3]?\s+-m\s+sglang\.launch_server\s+[^\n]+)",
            r"(python[3]?\s+.*launch_server\.py\s+[^\n]+)",
            r"(sglang\.launch_server\s+[^\n]+)",
        ]

        raw_command = None
        for pattern in patterns:
            match = re.search(pattern, log_content, re.IGNORECASE)
            if match:
                raw_command = match.group(1).strip()
                break

        if not raw_command:
            return None

        cmd = NodeLaunchCommand(
            backend_type=self.backend_type,
            worker_type=worker_type,
            raw_command=raw_command,
        )

        # Parse SGLang server arguments
        arg_patterns = {
            "model_path": r"--model(?:-path)?[=\s]+([^\s]+)",
            "served_model_name": r"--served-model-name[=\s]+([^\s]+)",
            "tp_size": r"--tp-size[=\s]+(\d+)",
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

        for field, pattern in arg_patterns.items():
            match = re.search(pattern, raw_command)
            if match:
                value = match.group(1)
                # Convert to appropriate type
                if field in ("tp_size", "dp_size", "ep_size", "port", "max_num_seqs", "max_model_len"):
                    value = int(value)
                elif field == "gpu_memory_utilization":
                    value = float(value)
                setattr(cmd, field, value)

        return cmd


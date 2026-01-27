# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRTLLM node log parser.
Parses logs from TensorRT-LLM workers launched via dynamo.trtllm.
Example log format:
    [33mRank0 run python3 -m dynamo.trtllm --model-path /model --served-model-name dsr1-fp8 ...
    Initializing the worker with config: Config(namespace=dynamo, component=prefill, ...)
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


@register_node_parser("trtllm")
class TRTLLMNodeParser:
    """Parser for TensorRT-LLM node logs.
    Parses logs from TRTLLM workers, including:
    - Launch command from dynamo.trtllm
    - Worker configuration from Config() dump
    - MPI rank and world size information
    
    Timestamp format: MM/DD/YYYY-HH:MM:SS (e.g., 01/23/2026-08:04:38)
    """

    @property
    def backend_type(self) -> str:
        return "trtllm"
    
    @staticmethod
    def parse_timestamp(timestamp: str) -> datetime:
        """Parse TRTLLM timestamp format to datetime object.
        
        Args:
            timestamp: Timestamp string in format MM/DD/YYYY-HH:MM:SS
            
        Returns:
            datetime object
            
        Raises:
            ValueError: If timestamp format is invalid
        """
        return datetime.strptime(timestamp, "%m/%d/%Y-%H:%M:%S")

    def parse_logs(self, log_dir: Path) -> list[NodeInfo]:
        """Parse all TRTLLM node logs in a directory.
        Args:
            log_dir: Directory containing *_prefill_*.out, *_decode_*.out files
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

        logger.info("Parsed %d TRTLLM node log files from %s", len(nodes), log_dir)
        return nodes

    def parse_single_log(self, log_path: Path) -> NodeInfo | None:
        """Parse a single TRTLLM log file.
        Args:
            log_path: Path to a node log file
        Returns:
            NodeInfo object or None if parsing failed
        """
        node_info = self._extract_node_info_from_filename(str(log_path))
        if not node_info:
            logger.warning("Could not extract node info from filename: %s", log_path)
            return None

        batches = []
        memory_snapshots = []
        config = {}
        launch_command = None

        try:
            # Handle encoding issues gracefully
            content = log_path.read_text(errors="replace")
            clean_content = ANSI_ESCAPE.sub("", content)

            # Parse launch command
            launch_command = self.parse_launch_command(clean_content, node_info["worker_type"])

            # Extract MPI configuration
            mpi_size_match = re.search(r"tllm_mpi_size:\s*(\d+)", clean_content)
            if mpi_size_match:
                config["mpi_world_size"] = int(mpi_size_match.group(1))

            # Extract TP/PP from Config() dump
            config_match = re.search(r"Config\((.*?)\)", clean_content)
            if config_match:
                config_str = config_match.group(1)

                tp_match = re.search(r"tensor_parallel_size=(\d+)", config_str)
                if tp_match:
                    config["tp_size"] = int(tp_match.group(1))

                pp_match = re.search(r"pipeline_parallel_size=(\d+)", config_str)
                if pp_match:
                    config["pp_size"] = int(pp_match.group(1))

                ep_match = re.search(r"expert_parallel_size=(\d+)", config_str)
                if ep_match:
                    config["ep_size"] = int(ep_match.group(1))

                max_batch_match = re.search(r"max_batch_size=(\d+)", config_str)
                if max_batch_match:
                    config["max_batch_size"] = int(max_batch_match.group(1))

                max_tokens_match = re.search(r"max_num_tokens=(\d+)", config_str)
                if max_tokens_match:
                    config["max_num_tokens"] = int(max_tokens_match.group(1))

                max_seq_match = re.search(r"max_seq_len=(\d+)", config_str)
                if max_seq_match:
                    config["max_seq_len"] = int(max_seq_match.group(1))

            # Extract from separate trtllm_config YAML references
            yaml_match = re.search(r"extra_engine_args=([^\s,]+\.yaml)", clean_content)
            if yaml_match:
                config["extra_engine_args"] = yaml_match.group(1)

            # Also extract from TensorRT-LLM engine args line which has actual parallelism
            engine_args_match = re.search(r"TensorRT-LLM engine args:\s*\{([^}]+)", clean_content)
            if engine_args_match:
                engine_str = engine_args_match.group(1)

                engine_tp_match = re.search(r"'tensor_parallel_size':\s*(\d+)", engine_str)
                if engine_tp_match:
                    config["tp_size"] = int(engine_tp_match.group(1))

                engine_pp_match = re.search(r"'pipeline_parallel_size':\s*(\d+)", engine_str)
                if engine_pp_match:
                    config["pp_size"] = int(engine_pp_match.group(1))

                engine_ep_match = re.search(r"'moe_expert_parallel_size':\s*(\d+)", engine_str)
                if engine_ep_match:
                    config["ep_size"] = int(engine_ep_match.group(1))

                engine_batch_match = re.search(r"'max_batch_size':\s*(\d+)", engine_str)
                if engine_batch_match:
                    config["max_batch_size"] = int(engine_batch_match.group(1))

                engine_tokens_match = re.search(r"'max_num_tokens':\s*(\d+)", engine_str)
                if engine_tokens_match:
                    config["max_num_tokens"] = int(engine_tokens_match.group(1))

                engine_seq_match = re.search(r"'max_seq_len':\s*(\d+)", engine_str)
                if engine_seq_match:
                    config["max_seq_len"] = int(engine_seq_match.group(1))

            # Parse iteration logs for batch metrics
            # Format: iter = X, ... num_scheduled_requests: X, states = {'num_ctx_requests': X, 'num_ctx_tokens': X, 'num_generation_tokens': X}
            batches = self._parse_iteration_logs(clean_content, node_info.get("worker_type", "unknown"))

            # Parse memory info
            memory_snapshots = self._parse_memory_info(clean_content)

        except Exception as e:
            logger.error("Error parsing %s: %s", log_path, e)
            return None

        logger.debug("Parsed %s: %d batches, %d memory snapshots, config=%s", log_path, len(batches), len(memory_snapshots), config)

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

    def _parse_iteration_logs(self, content: str, worker_type: str) -> list[BatchMetrics]:
        """Parse TRTLLM iteration logs for batch metrics.
        Format:
            [01/16/2026-06:20:17] [TRT-LLM] [RANK 0] [I] iter = 5559, ..., num_scheduled_requests: 1,
            states = {'num_ctx_requests': 0, 'num_ctx_tokens': 0, 'num_generation_tokens': 3}
        Args:
            content: Log file content (ANSI stripped)
            worker_type: Worker type (prefill, decode)
        Returns:
            List of BatchMetrics objects
        """
        batches = []

        # Pattern to match TRTLLM iteration logs
        iter_pattern = re.compile(
            r"\[(\d{2}/\d{2}/\d{4}-\d{2}:\d{2}:\d{2})\].*"
            r"iter\s*=\s*(\d+).*"
            r"num_scheduled_requests:\s*(\d+).*"
            r"states\s*=\s*\{([^}]+)\}"
        )

        for match in iter_pattern.finditer(content):
            timestamp = match.group(1)
            iteration = int(match.group(2))
            num_scheduled = int(match.group(3))
            states_str = match.group(4)

            # Parse states dict
            ctx_requests = 0
            ctx_tokens = 0
            gen_tokens = 0

            ctx_req_match = re.search(r"'num_ctx_requests':\s*(\d+)", states_str)
            if ctx_req_match:
                ctx_requests = int(ctx_req_match.group(1))

            ctx_tok_match = re.search(r"'num_ctx_tokens':\s*(\d+)", states_str)
            if ctx_tok_match:
                ctx_tokens = int(ctx_tok_match.group(1))

            gen_tok_match = re.search(r"'num_generation_tokens':\s*(\d+)", states_str)
            if gen_tok_match:
                gen_tokens = int(gen_tok_match.group(1))

            # Determine batch type based on content
            if ctx_tokens > 0:
                batch_type = "prefill"
            elif gen_tokens > 0:
                batch_type = "decode"
            else:
                batch_type = worker_type

            # Parse step time if available
            step_time = None
            step_match = re.search(r"host_step_time\s*=\s*([\d.]+)ms", match.group(0))
            if step_match:
                step_time = float(step_match.group(1))

            # Compute throughput (tokens/s)
            input_throughput = None
            gen_throughput = None
            if step_time and step_time > 0:
                if batch_type == "prefill" and ctx_tokens > 0:
                    # Prefill throughput: context tokens / step time
                    input_throughput = (ctx_tokens * 1000.0) / step_time
                elif batch_type == "decode" and gen_tokens > 0:
                    # Decode throughput: generation tokens / step time
                    gen_throughput = (gen_tokens * 1000.0) / step_time

            batches.append(
                BatchMetrics(
                    timestamp=timestamp,
                    dp=0,
                    tp=0,
                    ep=0,
                    batch_type=batch_type,
                    running_req=num_scheduled,
                    new_token=ctx_tokens if batch_type == "prefill" else None,
                    num_tokens=gen_tokens if batch_type == "decode" else None,
                    input_throughput=input_throughput,
                    gen_throughput=gen_throughput,
                )
            )

        return batches

    def _parse_memory_info(self, content: str) -> list[MemoryMetrics]:
        """Parse TRTLLM memory information.
        Format:
            Peak memory during memory usage profiling (torch + non-torch): 91.46 GiB,
            available KV cache memory when calculating max tokens: 41.11 GiB,
            fraction is set 0.85, kv size is 35136. device total memory 139.81 GiB
        Args:
            content: Log file content (ANSI stripped)
        Returns:
            List of MemoryMetrics objects
        """
        memory_snapshots = []

        # Pattern to match memory info
        mem_pattern = re.compile(
            r"\[(\d{2}/\d{2}/\d{4}-\d{2}:\d{2}:\d{2})\].*"
            r"Peak memory.*?:\s*([\d.]+)\s*GiB.*?"
            r"available KV cache memory.*?:\s*([\d.]+)\s*GiB.*?"
            r"device total memory\s*([\d.]+)\s*GiB"
        )

        for match in mem_pattern.finditer(content):
            timestamp = match.group(1)
            peak_mem = float(match.group(2))
            avail_kv = float(match.group(3))
            total_mem = float(match.group(4))

            memory_snapshots.append(
                MemoryMetrics(
                    timestamp=timestamp,
                    dp=0,
                    tp=0,
                    ep=0,
                    metric_type="memory",
                    mem_usage_gb=peak_mem,
                    avail_mem_gb=total_mem - peak_mem,
                    kv_cache_gb=avail_kv,
                )
            )

        # Also parse KV cache allocation info
        kv_alloc_pattern = re.compile(
            r"\[MemUsageChange\] Allocated\s*([\d.]+)\s*GiB for max tokens.*?\((\d+)\)"
        )

        for match in kv_alloc_pattern.finditer(content):
            kv_gb = float(match.group(1))
            max_tokens = int(match.group(2))

            memory_snapshots.append(
                MemoryMetrics(
                    timestamp="",
                    dp=0,
                    tp=0,
                    ep=0,
                    metric_type="kv_cache",
                    kv_cache_gb=kv_gb,
                    kv_tokens=max_tokens,
                )
            )

        return memory_snapshots

    def _extract_node_info_from_filename(self, filename: str) -> dict | None:
        """Extract node name and worker info from filename.
        Example: worker-0_prefill_w0.out
        Returns: {'node': 'worker-0', 'worker_type': 'prefill', 'worker_id': 'w0'}
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
        """Parse the TRTLLM worker launch command from log content.
        Looks for command lines like:
            [CMD] python3 -m dynamo.trtllm --model-path /model --served-model-name dsr1-fp8 --disaggregation-mode prefill
            python3 -m dynamo.trtllm --model-path /model --served-model-name dsr1-fp8 --disaggregation-mode prefill
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

        # Fallback: pattern to match TRTLLM launch commands (dynamo.trtllm or tensorrt_llm.serve)
        if not raw_command:
            patterns = [
                r"(?:Rank\d+\s+run\s+)?(python[3]?\s+-m\s+dynamo\.trtllm\s+[^\n]+)",
                r"(?:Rank\d+\s+run\s+)?(python[3]?\s+-m\s+tensorrt_llm\.serve\s+[^\n]+)",
                r"(trtllm-serve\s+[^\n]+)",
                r"(mpirun\s+.*trtllm[^\n]+)",
            ]

            for pattern in patterns:
                match = re.search(pattern, clean_content)
                if match:
                    raw_command = match.group(1).strip()
                    # Remove trailing "in background" if present
                    raw_command = re.sub(r"\s+in\s+background$", "", raw_command)
                    break

        if not raw_command:
            return None

        extra_args: dict[str, Any] = {}

        # Parse dynamo.trtllm / tensorrt_llm server arguments from command line
        arg_patterns = {
            "model_path": r"--model-path[=\s]+([^\s]+)",
            "served_model_name": r"--served-model-name[=\s]+([^\s]+)",
            "disaggregation_mode": r"--disaggregation-mode[=\s]+([^\s]+)",
            "host": r"--host[=\s]+([^\s]+)",
            "port": r"--port[=\s]+(\d+)",
        }

        for field, pattern in arg_patterns.items():
            match = re.search(pattern, raw_command)
            if match:
                value = match.group(1)
                if field == "port":
                    value = int(value)
                extra_args[field] = value

        # Also extract from TensorRT-LLM engine args if available (has actual parallelism values)
        engine_args_match = re.search(r"TensorRT-LLM engine args:\s*\{([^}]+)", clean_content)
        if engine_args_match:
            engine_str = engine_args_match.group(1)

            engine_patterns = {
                "tp_size": r"'tensor_parallel_size':\s*(\d+)",
                "pp_size": r"'pipeline_parallel_size':\s*(\d+)",
                "max_num_seqs": r"'max_batch_size':\s*(\d+)",
                "max_model_len": r"'max_seq_len':\s*(\d+)",
            }

            for field, pattern in engine_patterns.items():
                if field not in extra_args:
                    match = re.search(pattern, engine_str)
                    if match:
                        extra_args[field] = int(match.group(1))

        # Fallback to Config() dump
        if "tp_size" not in extra_args:
            config_match = re.search(r"Config\((.*?)\)", clean_content)
            if config_match:
                config_str = config_match.group(1)

                config_patterns = {
                    "tp_size": r"tensor_parallel_size=(\d+)",
                    "pp_size": r"pipeline_parallel_size=(\d+)",
                    "max_num_seqs": r"max_batch_size=(\d+)",
                    "max_model_len": r"max_seq_len=(\d+)",
                }

                for field, pattern in config_patterns.items():
                    if field not in extra_args:
                        match = re.search(pattern, config_str)
                        if match:
                            extra_args[field] = int(match.group(1))

        return NodeLaunchCommand(
            backend_type=self.backend_type,
            worker_type=worker_type,
            raw_command=raw_command,
            extra_args=extra_args,
        )
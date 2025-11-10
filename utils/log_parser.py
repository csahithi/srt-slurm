"""
Log file parser for extracting node-level metrics from .err files
"""

import logging
import os
import re
from typing import TypedDict

# Configure logging
logger = logging.getLogger(__name__)


# Type definitions for parsed metrics
class BatchMetrics(TypedDict, total=False):
    """Metrics from a prefill or decode batch log line."""

    timestamp: str
    dp: int
    tp: int
    ep: int
    type: str
    # Prefill-specific metrics
    new_seq: int
    new_token: int
    cached_token: int
    token_usage: float
    running_req: int
    queue_req: int
    prealloc_req: int
    inflight_req: int
    input_throughput: float
    # Decode-specific metrics
    gen_throughput: float
    transfer_req: int
    num_tokens: int
    preallocated_usage: float


class MemoryMetrics(TypedDict, total=False):
    """Memory metrics from log lines."""

    timestamp: str
    dp: int
    tp: int
    ep: int
    type: str
    avail_mem_gb: float
    mem_usage_gb: float
    kv_cache_gb: float
    kv_tokens: int


class ParsedErrFile(TypedDict, total=False):
    """Structure returned by parse_err_file.

    Contains runtime metrics from .err log files.
    Relevant for both prefill and decode nodes.
    """

    node_info: dict[str, str]  # Node name, worker type, worker ID
    prefill_batches: list[BatchMetrics]  # Batch processing metrics (primarily prefill)
    memory_snapshots: list[MemoryMetrics]  # Memory usage over time
    config: dict[str, int]  # TP/DP/EP configuration extracted from command line


def parse_dp_tp_ep_tag(line: str) -> tuple[int | None, int | None, int | None, str | None]:
    """Extract DP, TP, EP indices and timestamp from log line.

    Supports two formats:
    - Full: [2025-11-04 05:31:43 DP0 TP0 EP0]
    - Simple: [2025-11-04 07:05:55 TP0] (defaults DP=0, EP=0)

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

    return None, None, None, None


def parse_prefill_batch_line(line: str) -> dict | None:
    """Parse prefill batch log line for metrics.

    Example line:
    [2025-11-04 05:31:43 DP0 TP0 EP0] Prefill batch, #new-seq: 18, #new-token: 16384,
    #cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 0,
    #prealloc-req: 0, #inflight-req: 0, input throughput (token/s): 0.00,
    """
    dp, tp, ep, timestamp = parse_dp_tp_ep_tag(line)
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


def parse_decode_batch_line(line: str) -> dict | None:
    """Parse decode batch log line for metrics.

    Example line:
    [2025-11-04 05:32:32 DP31 TP31 EP31] Decode batch, #running-req: 7, #token: 7040,
    token usage: 0.00, pre-allocated usage: 0.00, #prealloc-req: 0, #transfer-req: 0,
    #retracted-req: 0, cuda graph: True, gen throughput (token/s): 6.73, #queue-req: 0,
    """
    dp, tp, ep, timestamp = parse_dp_tp_ep_tag(line)
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


def parse_memory_line(line: str) -> dict | None:
    """Parse memory-related log lines.

    Examples:
    [2025-11-04 05:27:13 DP0 TP0 EP0] Load weight end. type=DeepseekV3ForCausalLM,
    dtype=torch.bfloat16, avail mem=75.11 GB, mem usage=107.07 GB.

    [2025-11-04 05:27:13 DP0 TP0 EP0] KV Cache is allocated. #tokens: 524288, KV size: 17.16 GB
    """
    dp, tp, ep, timestamp = parse_dp_tp_ep_tag(line)
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


def extract_node_info_from_filename(filename: str) -> dict | None:
    """Extract node name and worker info from filename.

    Example: watchtower-navy-cn01_prefill_w0.err
    Returns: {'node': 'watchtower-navy-cn01', 'worker_type': 'prefill', 'worker_id': 'w0'}
    """
    match = re.match(r"([^_]+)_(prefill|decode|frontend)_([^.]+)\.err", os.path.basename(filename))
    if match:
        return {"node": match.group(1), "worker_type": match.group(2), "worker_id": match.group(3)}
    return None


def parse_err_file(filepath: str) -> ParsedErrFile | None:
    """Parse a single .err file and extract all metrics.

    Expected filename format: <node>_<service>_<id>.err
    Expected content: Runtime logs with DP/TP/EP tags

    Args:
        filepath: Path to the .err file

    Returns:
        {
            'node_info': {...},
            'prefill_batches': [...],
            'memory_snapshots': [...],
            'kv_cache_details': [...],
            'transfer_engine_metrics': [...],
            'config': {...}  # TP/DP/EP configuration
        }
        or None if file cannot be parsed
    """
    node_info = extract_node_info_from_filename(filepath)
    if not node_info:
        logger.warning(
            f"Could not extract node info from filename: {filepath}. "
            f"Expected format: <node>_<service>_<id>.err"
        )
        return None

    prefill_batches = []
    memory_snapshots = []
    config = {}

    try:
        with open(filepath) as f:
            for line in f:
                # Parse prefill batch metrics
                batch_metrics = parse_prefill_batch_line(line)
                if batch_metrics:
                    prefill_batches.append(batch_metrics)

                # Parse decode batch metrics
                decode_metrics = parse_decode_batch_line(line)
                if decode_metrics:
                    prefill_batches.append(decode_metrics)  # Store both in same list

                # Parse memory metrics
                mem_metrics = parse_memory_line(line)
                if mem_metrics:
                    memory_snapshots.append(mem_metrics)

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
        logger.error(f"Error parsing {filepath}: {e}")
        return None

    # Validation: Log if we found no metrics
    total_metrics = len(prefill_batches) + len(memory_snapshots)

    if total_metrics == 0:
        logger.warning(
            f"Parsed {filepath} but found no metrics. "
            f"Expected to find lines with DP/TP/EP tags. "
            f"Log format may have changed."
        )

    logger.debug(
        f"Parsed {filepath}: {len(prefill_batches)} batches, "
        f"{len(memory_snapshots)} memory snapshots"
    )

    return {
        "node_info": node_info,
        "prefill_batches": prefill_batches,
        "memory_snapshots": memory_snapshots,
        "config": config,
    }


def parse_all_err_files(run_path: str) -> list[ParsedErrFile]:
    """Parse all .err files in a run directory.

    Args:
        run_path: Path to the run directory containing .err files

    Returns:
        List of parsed node metrics
    """
    err_files: list[ParsedErrFile] = []

    if not os.path.exists(run_path):
        logger.error(f"Run path does not exist: {run_path}")
        return err_files

    total_err_files = 0
    parsed_successfully = 0

    for file in os.listdir(run_path):
        if file.endswith(".err") and ("prefill" in file or "decode" in file):
            total_err_files += 1
            filepath = os.path.join(run_path, file)
            parsed = parse_err_file(filepath)
            if parsed:
                err_files.append(parsed)
                parsed_successfully += 1

    logger.info(
        f"Parsed {parsed_successfully}/{total_err_files} prefill/decode .err files from {run_path}"
    )

    if total_err_files == 0:
        logger.warning(f"No prefill/decode .err files found in {run_path}")

    return err_files


def get_node_label(node_data: dict) -> str:
    """Generate a display label for a node with its configuration.

    Example: "[3320] cn01-prefill-w0 (DP0-3, TP8, EP8)"
    """
    node_info = node_data["node_info"]
    config = node_data["config"]
    run_id = node_data.get("run_id", "")

    node_name = node_info["node"].replace("watchtower-navy-", "")
    worker = f"{node_info['worker_type']}-{node_info['worker_id']}"

    # Get DP range if we have batch data
    dp_indices = set()
    for batch in node_data.get("prefill_batches", []):
        if "dp" in batch:
            dp_indices.add(batch["dp"])

    if dp_indices:
        dp_min, dp_max = min(dp_indices), max(dp_indices)
        if dp_min == dp_max:
            dp_str = f"DP{dp_min}"
        else:
            dp_str = f"DP{dp_min}-{dp_max}"
    else:
        dp_str = "DP?"

    tp_str = f"TP{config.get('tp_size', '?')}"
    ep_str = f"EP{config.get('ep_size', '?')}"

    # Include run_id prefix if available (for multi-run comparisons)
    if run_id:
        # Extract just the job number from directory name like "3320_1P_4D_20251104_231843"
        run_prefix = run_id.split("_")[0] if "_" in run_id else run_id
        return f"[{run_prefix}] {node_name}-{worker} ({dp_str}, {tp_str}, {ep_str})"
    else:
        return f"{node_name}-{worker} ({dp_str}, {tp_str}, {ep_str})"

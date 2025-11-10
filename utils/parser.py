"""
Parser module for benchmark logs - adapted from parse.py
"""

import json
import os
import re


def extract_job_id(dirname: str) -> int:
    """Extract job ID from directory name for sorting.

    Handles formats like:
    - 12345_3P_1D_20250104_123456 (disaggregated)
    - 12345_4A_20250104_123456 (aggregated)
    - 12345 (legacy format)
    """
    try:
        return int(dirname.split("_")[0])
    except (ValueError, IndexError):
        return -1


def analyze_sgl_out(folder: str) -> dict:
    """Analyze SGLang/vLLM benchmark output files."""
    result = []
    for file in os.listdir(folder):
        filepath = os.path.join(folder, file)
        with open(filepath) as f:
            content = json.load(f)
            res = [
                content["max_concurrency"],
                content["output_throughput"],
                content["mean_itl_ms"],
                content["mean_ttft_ms"],
                content["request_rate"],
            ]

            if "mean_tpot_ms" in content:
                res.append(content["mean_tpot_ms"])
            result.append(res)

    out = {
        "request_rate": [],
        "concurrencies": [],
        "output_tps": [],
        "mean_itl_ms": [],
        "mean_ttft_ms": [],
        "mean_tpot_ms": [],
    }

    for data in sorted(result, key=lambda x: x[0]):
        con, tps, itl, ttft, req_rate = data[0:5]
        out["concurrencies"].append(con)
        out["output_tps"].append(tps)
        out["mean_itl_ms"].append(itl)
        out["mean_ttft_ms"].append(ttft)
        out["request_rate"].append(req_rate)

        if len(data) >= 6:
            out["mean_tpot_ms"].append(data[5])

    return out


def analyze_gap_out(folder: str) -> dict:
    """Analyze GAP benchmark output files."""
    result = []
    for file in os.listdir(folder):
        filepath = os.path.join(folder, file)
        with open(filepath) as f:
            content = json.load(f)
            result.append(
                (
                    content["input_config"]["perf_analyzer"]["stimulus"]["concurrency"],
                    content["output_token_throughput_per_user"]["avg"],
                    content["output_token_throughput"]["avg"],
                )
            )

    out = {"concurrencies": [], "output_tps": [], "output_tps_per_user": []}

    for con, tpspuser, tps in sorted(result, key=lambda x: x[0]):
        out["concurrencies"].append(con)
        out["output_tps"].append(tps)
        out["output_tps_per_user"].append(tpspuser)

    return out


def count_nodes_and_gpus(path: str) -> tuple[dict, dict, list]:
    """Count prefill nodes, decode nodes, and frontends from log files."""
    files = os.listdir(path)

    prefill_nodes = {}
    decode_nodes = {}
    frontends = []

    for file in files:
        p_re = re.search(
            r"([-_A-Za-z0-9]+)_(prefill|decode|nginx|frontend)_([a-zA-Z0-9]+).out", file
        )
        if p_re is not None:
            _, node_type, number = p_re.groups()
            if node_type == "prefill":
                if number not in prefill_nodes:
                    prefill_nodes[number] = []
                prefill_nodes[number].append(file)
            elif node_type == "decode":
                if number not in decode_nodes:
                    decode_nodes[number] = []
                decode_nodes[number].append(file)
            elif node_type == "frontend":
                frontends.append(file)

    return prefill_nodes, decode_nodes, frontends


def parse_run_date(dirname: str) -> str | None:
    """Parse date from run directory name.

    Expected format: <jobid>_<config>_YYYYMMDD_HHMMSS
    Example: 3262_3P_1D_20251104_051714

    Returns:
        Formatted date string like "2025-11-04 05:17:14" or None
    """
    try:
        parts = dirname.split("_")
        if len(parts) >= 2:
            # Look for date pattern (8 digits)
            for i, part in enumerate(parts):
                if len(part) == 8 and part.isdigit():
                    # Found date, check if next part is time
                    date_str = part
                    time_str = (
                        parts[i + 1] if i + 1 < len(parts) and len(parts[i + 1]) == 6 else "000000"
                    )

                    # Parse YYYYMMDD
                    year = date_str[0:4]
                    month = date_str[4:6]
                    day = date_str[6:8]

                    # Parse HHMMSS
                    hour = time_str[0:2]
                    minute = time_str[2:4]
                    second = time_str[4:6]

                    return f"{year}-{month}-{day} {hour}:{minute}:{second}"
    except Exception:
        pass
    return None


def parse_topology_from_dirname(dirname: str) -> tuple[int | None, int | None]:
    """Parse topology (XP_YD) from run directory name.

    Expected format: <jobid>_XP_YD_YYYYMMDD_HHMMSS
    Example: 3274_1P_4D_20251104_065031 -> (1, 4)

    Returns:
        Tuple of (prefill_workers, decode_workers) or (None, None) if not found
    """
    try:
        # Match pattern like 1P_4D or 3P_1D
        match = re.search(r"_(\d+)P_(\d+)D_", dirname)
        if match:
            prefill_workers = int(match.group(1))
            decode_workers = int(match.group(2))
            return (prefill_workers, decode_workers)
    except Exception:
        pass
    return (None, None)


def parse_container_image(run_path: str) -> str | None:
    """Parse container image from log.err or log.out files.

    Looks for patterns like:
    - CONTAINER_IMAGE=/path/to/container.sqsh
    - --container-image=/path/to/container.sqsh

    Args:
        run_path: Path to run directory

    Returns:
        Cleaned container name like "sglang+v0.5.4.post2-dyn" or None
    """
    # Check log.err first, then log.out
    for log_file in ["log.err", "log.out"]:
        log_path = os.path.join(run_path, log_file)
        if not os.path.exists(log_path):
            continue

        try:
            with open(log_path) as f:
                # Read first 100 lines (container info is usually at the top)
                for i, line in enumerate(f):
                    if i > 100:
                        break

                    # Look for CONTAINER_IMAGE= or --container-image=
                    match = re.search(r"(?:CONTAINER_IMAGE=|--container-image=)(.+\.sqsh)", line)
                    if match:
                        container_path = match.group(1)

                        # Extract just the filename
                        container_filename = os.path.basename(container_path)

                        # Remove .sqsh extension
                        container_name = container_filename.replace(".sqsh", "")

                        # Clean up: remove username prefix if present
                        # e.g., "ishandhanani+sglang+v0.5.4.post2-dyn" -> "sglang+v0.5.4.post2-dyn"
                        if "+" in container_name:
                            parts = container_name.split("+", 1)
                            if len(parts) > 1:
                                container_name = parts[1]

                        return container_name
        except Exception:
            pass

    return None


def analyze_run(path: str) -> dict:
    """Analyze a single benchmark run directory."""
    files = os.listdir(path)

    prefill_nodes, decode_nodes, frontends = count_nodes_and_gpus(path)

    profile_result = {}

    for file in files:
        profiler_match = re.match(r"(sglang|vllm|gap)_isl_([0-9]+)_osl_([0-9]+)", file)
        if profiler_match:
            profiler, isl, osl = profiler_match.groups()
            folder_path = os.path.join(path, file)

            if profiler == "gap":
                profile_result = analyze_gap_out(folder_path)
            else:
                profile_result = analyze_sgl_out(folder_path)

            profile_result["profiler_type"] = profiler
            profile_result["isl"] = isl
            profile_result["osl"] = osl

    # Extract date from directory name
    dirname = os.path.basename(path)
    run_date = parse_run_date(dirname)

    # Extract topology from directory name (XP_YD)
    prefill_workers, decode_workers = parse_topology_from_dirname(dirname)

    # Extract container image from log files
    container = parse_container_image(path)

    config = {"slurm_job_id": dirname, "path": path, "run_date": run_date, "container": container}

    # Use topology from folder name if available, otherwise fall back to counting files
    if prefill_workers is not None:
        config["prefill_dp"] = prefill_workers
    elif len(prefill_nodes.values()) != 0:
        config["prefill_dp"] = len(prefill_nodes.keys())

    if decode_workers is not None:
        config["decode_dp"] = decode_workers
    elif len(decode_nodes.values()) != 0:
        config["decode_dp"] = len(decode_nodes.keys())

    # Still compute TP from files
    if len(prefill_nodes.values()) != 0:
        config["prefill_tp"] = len(list(prefill_nodes.values())[0]) * 4

    if len(decode_nodes.values()) != 0:
        config["decode_tp"] = len(list(decode_nodes.values())[0]) * 4

    if len(frontends) != 0:
        config["frontends"] = len(frontends)

    result = {**config, **profile_result}
    return result


def find_all_runs(logs_dir: str) -> list[dict]:
    """Find and analyze all benchmark runs in the logs directory."""
    paths = [
        os.path.join(logs_dir, x)
        for x in os.listdir(logs_dir)
        if ".py" not in x and os.path.isdir(os.path.join(logs_dir, x))
    ]

    all_runs = []
    for path in sorted(paths, key=lambda p: extract_job_id(os.path.basename(p)), reverse=True):
        try:
            result = analyze_run(path)
            if "output_tps" in result and result["output_tps"]:  # Only include runs with data
                all_runs.append(result)
        except Exception as e:
            print(f"Error parsing {path}: {e}")
            continue

    return all_runs

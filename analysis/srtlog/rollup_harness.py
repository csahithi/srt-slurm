#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Rollup harness for batch processing experiment logs.

Recursively searches for sbatch_script.sh files and runs rollup on each job directory.

Usage:
    python -m analysis.srtlog.rollup_harness --log-dir /path/to/outputs
    python -m analysis.srtlog.rollup_harness --log-dir /path/to/outputs --dry-run
    python -m analysis.srtlog.rollup_harness --log-dir /path/to/outputs --output-dir /path/to/rollups
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def find_job_directories(log_dir: Path) -> list[Path]:
    """Find all job directories by searching for sbatch_script.sh files.

    Args:
        log_dir: Root directory to search

    Returns:
        List of job directory paths (parent dirs of sbatch_script.sh)
    """
    job_dirs = []
    for sbatch_script in log_dir.rglob("sbatch_script.sh"):
        job_dir = sbatch_script.parent
        job_dirs.append(job_dir)

    # Sort by job ID (directory name) if numeric
    job_dirs.sort(key=lambda p: (int(p.name) if p.name.isdigit() else p.name))
    return job_dirs


def load_job_config(job_dir: Path) -> dict[str, Any] | None:
    """Load job configuration from config.yaml.

    Args:
        job_dir: Job directory containing config.yaml

    Returns:
        Parsed config dict or None if not found
    """
    config_path = job_dir / "config.yaml"
    if not config_path.exists():
        return None

    try:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning("Failed to load %s: %s", config_path, e)
        return None


def get_log_dir(job_dir: Path) -> Path | None:
    """Get the logs directory for a job.

    Args:
        job_dir: Job directory

    Returns:
        Path to logs directory or None if not found
    """
    logs_dir = job_dir / "logs"
    if logs_dir.exists():
        return logs_dir
    return None


def _add_launch_command_to_node(node_rollup: Any, node_parser: Any, logs_dir: Path) -> None:
    """Parse and add launch command to a node rollup.

    Args:
        node_rollup: NodeRollup object to update
        node_parser: Node parser with parse_launch_command method
        logs_dir: Directory containing log files
    """
    from srtctl.cli.mixins.rollup import LaunchCommandRollup

    node_name = node_rollup.node_name
    worker_type = node_rollup.worker_type
    worker_id = node_rollup.worker_id

    # Try different filename patterns
    patterns = [
        f"{node_name}_{worker_type}_{worker_id}",  # worker-4_decode_w0
        f"worker-*_{worker_type}_{worker_id}",     # wildcard node
    ]

    for pattern in patterns:
        # Try both .out and .err files
        for ext in [".out", ".err"]:
            if "*" in pattern:
                # Glob pattern
                for log_file in logs_dir.glob(f"{pattern}{ext}"):
                    if _try_parse_launch_command(node_rollup, node_parser, log_file, worker_type):
                        return
            else:
                log_file = logs_dir / f"{pattern}{ext}"
                if log_file.exists():
                    if _try_parse_launch_command(node_rollup, node_parser, log_file, worker_type):
                        return


def _try_parse_launch_command(node_rollup: Any, node_parser: Any, log_file: Path, worker_type: str) -> bool:
    """Try to parse launch command from a log file.

    Args:
        node_rollup: NodeRollup object to update
        node_parser: Node parser with parse_launch_command method
        log_file: Log file to parse
        worker_type: Worker type (prefill, decode, agg)

    Returns:
        True if command was found and added
    """
    from srtctl.cli.mixins.rollup import LaunchCommandRollup

    try:
        content = log_file.read_text(errors="replace")
        cmd = node_parser.parse_launch_command(content, worker_type=worker_type)
        if cmd:
            args = cmd.extra_args
            node_rollup.launch_command = LaunchCommandRollup(
                raw_command=cmd.raw_command,
                command_type="worker",
                model_path=args.get("model_path"),
                served_model_name=args.get("served_model_name"),
                worker_type=worker_type,
                backend_type=cmd.backend_type,
                disaggregation_mode=args.get("disaggregation_mode"),
                tp_size=args.get("tp_size"),
                pp_size=args.get("pp_size"),
                dp_size=args.get("dp_size"),
                ep_size=args.get("ep_size"),
                port=args.get("port"),
                max_num_seqs=args.get("max_num_seqs"),
                max_model_len=args.get("max_model_len"),
            )
            logger.debug("Parsed launch command for %s from %s", node_rollup.node_name, log_file.name)
            return True
    except Exception as e:
        logger.debug("Failed to parse launch command from %s: %s", log_file, e)

    return False


def run_rollup_on_job(job_dir: Path, output_dir: Path | None = None) -> dict[str, Any] | None:
    """Run rollup on a single job directory.

    Args:
        job_dir: Job directory containing config.yaml and logs/
        output_dir: Optional output directory for rollup.json

    Returns:
        Rollup summary dict or None if failed
    """
    from analysis.srtlog.parsers import get_benchmark_parser, get_node_parser, list_benchmark_parsers, list_node_parsers
    from srtctl.cli.mixins.rollup import (
        EnvironmentConfig,
        LaunchCommandRollup,
        NodesSummary,
        RollupResult,
        RollupSummary,
    )

    job_id = job_dir.name
    logs_dir = get_log_dir(job_dir)

    if not logs_dir:
        logger.warning("No logs directory found in %s", job_dir)
        return None

    config = load_job_config(job_dir)
    if not config:
        logger.warning("No config.yaml found in %s", job_dir)
        return None

    # Extract config values
    backend_type = config.get("backend", {}).get("type", "unknown")
    benchmark_type = config.get("benchmark", {}).get("type", "sa-bench")
    model_name = config.get("model", {}).get("served_model_name", "unknown")
    
    resources = config.get("resources", {})
    is_disaggregated = resources.get("prefill_nodes") is not None
    
    logger.info("Processing job %s: backend=%s, benchmark=%s", job_id, backend_type, benchmark_type)

    # Parse benchmark results
    results = []
    try:
        parser = get_benchmark_parser(benchmark_type)
        
        # Find result directories
        for entry in logs_dir.iterdir():
            if entry.is_dir() and "_isl_" in entry.name and "_osl_" in entry.name:
                dir_results = parser.parse_result_directory(entry)
                results.extend(dir_results)
        
        # Also check for AIPerf results
        if hasattr(parser, "find_aiperf_results"):
            for aiperf_path in parser.find_aiperf_results(logs_dir):
                result = parser.parse_result_json(aiperf_path)
                if result.get("output_tps") is not None or result.get("output_throughput") is not None:
                    results.append(result)
                    
    except ValueError:
        logger.warning("No benchmark parser for %s, available: %s", benchmark_type, list_benchmark_parsers())
    except Exception as e:
        logger.warning("Failed to parse benchmark results: %s", e)

    if not results:
        logger.warning("No benchmark results found in %s", logs_dir)

    # Parse node metrics
    nodes_summary = None
    node_parser = None
    try:
        node_parser = get_node_parser(backend_type)
        nodes = node_parser.parse_logs(logs_dir)
        if nodes:
            nodes_summary = NodesSummary.from_node_metrics_list(nodes)
            logger.info("  Found %d nodes (%d prefill, %d decode)", 
                       len(nodes_summary.nodes),
                       nodes_summary.total_prefill_nodes,
                       nodes_summary.total_decode_nodes)
            
            # Parse launch commands for each node
            for node_rollup in nodes_summary.nodes:
                _add_launch_command_to_node(node_rollup, node_parser, logs_dir)
                
    except ValueError:
        logger.debug("No node parser for %s, available: %s", backend_type, list_node_parsers())
    except Exception as e:
        logger.warning("Failed to parse node metrics: %s", e)

    # Parse benchmark launch command
    benchmark_command = None
    benchmark_out = logs_dir / "benchmark.out"
    if benchmark_out.exists():
        try:
            parser = get_benchmark_parser(benchmark_type)
            cmd = parser.parse_launch_command(benchmark_out.read_text(errors="replace"))
            if cmd:
                benchmark_command = LaunchCommandRollup(
                    raw_command=cmd.raw_command,
                    command_type="benchmark",
                    model_path=cmd.model,
                    benchmark_type=cmd.benchmark_type,
                    base_url=cmd.base_url,
                    max_concurrency=cmd.max_concurrency,
                    input_len=cmd.input_len,
                    output_len=cmd.output_len,
                )
        except Exception as e:
            logger.debug("Failed to parse benchmark command: %s", e)

    # Parse environment config
    environment_config = None
    try:
        import yaml
        env_config = EnvironmentConfig()
        
        backend_section = config.get("backend", {})
        if "prefill_environment" in backend_section:
            env_config.prefill_environment = backend_section["prefill_environment"]
        if "decode_environment" in backend_section:
            env_config.decode_environment = backend_section["decode_environment"]
        if "aggregated_environment" in backend_section:
            env_config.aggregated_environment = backend_section["aggregated_environment"]
            
        # Load TRTLLM config files
        prefill_yaml = logs_dir / "trtllm_config_prefill.yaml"
        decode_yaml = logs_dir / "trtllm_config_decode.yaml"
        
        if prefill_yaml.exists():
            with open(prefill_yaml) as f:
                env_config.prefill_engine_config = yaml.safe_load(f)
        if decode_yaml.exists():
            with open(decode_yaml) as f:
                env_config.decode_engine_config = yaml.safe_load(f)
                
        if any([env_config.prefill_environment, env_config.decode_environment, 
                env_config.prefill_engine_config, env_config.decode_engine_config]):
            environment_config = env_config
            
    except Exception as e:
        logger.debug("Failed to parse environment config: %s", e)

    # Build rollup summary
    benchmark_config = config.get("benchmark", {})
    
    # Compute total GPUs
    if is_disaggregated:
        prefill_gpus = resources.get("prefill_nodes", 0) * resources.get("gpus_per_node", 8)
        decode_gpus = resources.get("decode_nodes", 0) * resources.get("gpus_per_node", 8)
        total_gpus = prefill_gpus + decode_gpus
    else:
        total_gpus = resources.get("agg_nodes", 1) * resources.get("gpus_per_node", 8)
        prefill_gpus = 0
        decode_gpus = 0

    summary = RollupSummary(
        job_id=job_id,
        job_name=config.get("name", "unknown"),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        model_path=config.get("model", {}).get("path", ""),
        model_name=model_name,
        precision=config.get("model", {}).get("precision", "unknown"),
        gpu_type=resources.get("gpu_type", "unknown"),
        gpus_per_node=resources.get("gpus_per_node", 8),
        backend_type=backend_type,
        frontend_type=config.get("frontend", {}).get("type", "unknown"),
        is_disaggregated=is_disaggregated,
        total_nodes=resources.get("prefill_nodes", 0) + resources.get("decode_nodes", 0) if is_disaggregated else resources.get("agg_nodes", 1),
        total_gpus=total_gpus,
        prefill_nodes=resources.get("prefill_nodes") if is_disaggregated else None,
        decode_nodes=resources.get("decode_nodes") if is_disaggregated else None,
        prefill_gpus=prefill_gpus if is_disaggregated else None,
        decode_gpus=decode_gpus if is_disaggregated else None,
        agg_nodes=resources.get("agg_nodes") if not is_disaggregated else None,
        benchmark_type=benchmark_type,
        isl=benchmark_config.get("isl"),
        osl=benchmark_config.get("osl"),
        concurrencies=benchmark_config.get("concurrencies", []),
        nodes_summary=nodes_summary,
        environment_config=environment_config,
        benchmark_command=benchmark_command,
        tags=config.get("tags", []),
    )

    # Convert results to RollupResult objects
    for data in results:
        result = RollupResult(
            concurrency=data.get("max_concurrency", 0),
            output_tps=data.get("output_throughput", 0) or data.get("output_tps", 0),
            total_tps=data.get("total_token_throughput"),
            request_throughput=data.get("request_throughput"),
            mean_ttft_ms=data.get("mean_ttft_ms"),
            mean_tpot_ms=data.get("mean_tpot_ms"),
            mean_itl_ms=data.get("mean_itl_ms"),
            mean_e2el_ms=data.get("mean_e2el_ms"),
            median_ttft_ms=data.get("median_ttft_ms"),
            median_itl_ms=data.get("median_itl_ms"),
            p99_ttft_ms=data.get("p99_ttft_ms"),
            p99_itl_ms=data.get("p99_itl_ms"),
            total_input_tokens=data.get("total_input_tokens"),
            total_output_tokens=data.get("total_output_tokens"),
            duration=data.get("duration"),
            completed=data.get("completed"),
            num_prompts=data.get("num_prompts"),
        )
        summary.results.append(result)

    # Compute summary stats
    summary.compute_summary_stats()

    # Write rollup
    if output_dir:
        rollup_path = output_dir / f"{job_id}_rollup.json"
    else:
        rollup_path = logs_dir / "rollup.json"

    rollup_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rollup_path, "w") as f:
        json.dump(asdict(summary), f, indent=2, default=str)

    logger.info("  Wrote rollup to %s", rollup_path)
    logger.info("  Results: %d, Max TPS: %.1f", len(summary.results), summary.max_output_tps or 0)

    return asdict(summary)


def main():
    parser = argparse.ArgumentParser(
        description="Rollup harness for batch processing experiment logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all jobs in outputs directory
    python -m analysis.srtlog.rollup_harness --log-dir /path/to/outputs

    # Dry run - just list jobs that would be processed
    python -m analysis.srtlog.rollup_harness --log-dir /path/to/outputs --dry-run

    # Write rollups to a separate directory
    python -m analysis.srtlog.rollup_harness --log-dir /path/to/outputs --output-dir /path/to/rollups

    # Process only specific jobs
    python -m analysis.srtlog.rollup_harness --log-dir /path/to/outputs --jobs 585 586 587
        """,
    )
    parser.add_argument("--log-dir", required=True, type=Path, help="Root directory to search for jobs")
    parser.add_argument("--output-dir", type=Path, help="Output directory for rollup files (default: in-place)")
    parser.add_argument("--dry-run", action="store_true", help="List jobs without processing")
    parser.add_argument("--jobs", nargs="+", help="Only process specific job IDs")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.log_dir.exists():
        logger.error("Log directory does not exist: %s", args.log_dir)
        sys.exit(1)

    # Find all job directories
    job_dirs = find_job_directories(args.log_dir)
    logger.info("Found %d job directories in %s", len(job_dirs), args.log_dir)

    # Filter by job IDs if specified
    if args.jobs:
        job_dirs = [d for d in job_dirs if d.name in args.jobs]
        logger.info("Filtered to %d jobs: %s", len(job_dirs), [d.name for d in job_dirs])

    if args.dry_run:
        print("\nJob directories found:")
        for job_dir in job_dirs:
            config = load_job_config(job_dir)
            if config:
                backend = config.get("backend", {}).get("type", "?")
                benchmark = config.get("benchmark", {}).get("type", "?")
                print(f"  {job_dir.name}: backend={backend}, benchmark={benchmark}")
            else:
                print(f"  {job_dir.name}: (no config.yaml)")
        return

    # Process each job
    successful = 0
    failed = 0
    
    for job_dir in job_dirs:
        try:
            result = run_rollup_on_job(job_dir, args.output_dir)
            if result:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            logger.error("Failed to process %s: %s", job_dir, e)
            failed += 1

    logger.info("Complete: %d successful, %d failed", successful, failed)


if __name__ == "__main__":
    main()


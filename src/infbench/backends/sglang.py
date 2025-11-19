#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SGLang backend support - config generation and command rendering.
"""

import json
import logging
import os
import tempfile
import yaml
from pathlib import Path
from typing import Any


def expand_template(template: Any, values: dict[str, Any]) -> Any:
    """Recursively expand template strings with values"""
    if isinstance(template, dict):
        return {k: expand_template(v, values) for k, v in template.items()}
    elif isinstance(template, list):
        return [expand_template(item, values) for item in template]
    elif isinstance(template, str):
        # Replace {param} with actual value
        result = template
        for key, value in values.items():
            placeholder = f"{{{key}}}"
            result = result.replace(placeholder, str(value))
        return result
    else:
        return template


def generate_sglang_config_file(user_config: dict, params: dict = None) -> Path:
    """
    Generate SGLang config YAML file from user config.

    Args:
        user_config: User's YAML config
        params: Sweep parameters to substitute (optional)

    Returns:
        Path to generated SGLang config file
    """
    if 'sglang_config' not in user_config.get('backend', {}):
        return None

    sglang_cfg = user_config['backend']['sglang_config']

    # Expand templates if sweeping
    if params:
        sglang_cfg = expand_template(sglang_cfg, params)
        logging.info(f"Expanded config with params: {params}")

    # Merge shared config into prefill/decode
    shared = sglang_cfg.get('shared', {})
    result = {}

    for mode in ['prefill', 'decode']:
        if mode in sglang_cfg:
            # Merge: shared + mode-specific (mode-specific wins on conflicts)
            result[mode] = {**shared, **sglang_cfg[mode]}

    # Write to temp file
    fd, temp_path = tempfile.mkstemp(suffix='.yaml', prefix='sglang_config_')
    with os.fdopen(fd, 'w') as f:
        yaml.dump(result, f, default_flow_style=False)

    logging.info(f"Generated SGLang config: {temp_path}")
    return Path(temp_path)


def render_sglang_command(config: dict, sglang_config_path: Path, mode: str = "prefill") -> str:
    """
    Render the full SGLang command that would be executed with all flags inlined.

    Args:
        config: User config dict
        sglang_config_path: Path to generated SGLang config
        mode: "prefill" or "decode"

    Returns:
        Multi-line string showing the full command with environment variables and all flags
    """
    backend = config.get('backend', {})
    resources = config.get('resources', {})

    # Environment variables
    env_vars = []
    if 'environment' in backend:
        for key, val in backend['environment'].items():
            env_vars.append(f"{key}={val}")

    # Add decode-specific env vars if in decode mode
    if mode == "decode" and 'decode_environment' in backend:
        for key, val in backend['decode_environment'].items():
            env_vars.append(f"{key}={val}")

    # Build command
    lines = []

    # Environment variables (one per line with backslash continuation)
    if env_vars:
        for env_var in env_vars:
            lines.append(f"{env_var} \\")

    # Python command
    lines.append("python3 -m dynamo.sglang \\")

    # Load the generated SGLang config and inline all flags
    with open(sglang_config_path) as f:
        sglang_config = yaml.load(f, Loader=yaml.FullLoader)

    # Get flags for this mode
    mode_config = sglang_config.get(mode, {})

    # Convert config dict to command-line flags
    for key, value in sorted(mode_config.items()):
        # Convert underscores to hyphens for CLI flags
        flag_name = key.replace('_', '-')

        # Handle different value types
        if isinstance(value, bool):
            if value:
                lines.append(f"    --{flag_name} \\")
        elif isinstance(value, list):
            # For lists, pass each value as separate argument
            values_str = " ".join(str(v) for v in value)
            lines.append(f"    --{flag_name} {values_str} \\")
        else:
            lines.append(f"    --{flag_name} {value} \\")

    # Determine nnodes based on mode and whether aggregated or disaggregated
    is_aggregated = 'agg_nodes' in resources
    if is_aggregated:
        nnodes = resources['agg_nodes']
    else:
        # Disaggregated: prefill and decode have different node counts
        if mode == "prefill":
            nnodes = resources['prefill_nodes']
        else:  # decode
            nnodes = resources['decode_nodes']

    # Coordination flags
    lines.append("    --dist-init-addr $HOST_IP_MACHINE:$PORT \\")
    lines.append(f"    --nnodes {nnodes} \\")
    lines.append("    --node-rank $RANK \\")

    # Parallelism flags (computed from resources)
    gpus_per_node = resources.get('gpus_per_node', 4)
    lines.append(f"    --ep-size {gpus_per_node} \\")
    lines.append(f"    --tp-size {gpus_per_node} \\")
    lines.append(f"    --dp-size {gpus_per_node}")

    return "\n".join(lines)


def yaml_to_args(config: dict, sglang_config_path: Path = None) -> list[str]:
    """Convert YAML config to submit_job_script.py arguments"""

    args = [
        "--job-name", config['name'],
        "--account", config['slurm']['account'],
        "--partition", config['slurm']['partition'],
        "--time-limit", config['slurm']['time_limit'],

        # Model
        "--model-dir", config['model']['path'],
        "--container-image", config['model']['container'],

        # Resources
        "--gpus-per-node", str(config['resources']['gpus_per_node']),
    ]

    # Mode: aggregated or disaggregated
    if 'agg_nodes' in config['resources']:
        # Aggregated mode
        args.extend([
            "--agg-nodes", str(config['resources']['agg_nodes']),
            "--agg-workers", str(config['resources']['agg_workers']),
        ])
    else:
        # Disaggregated mode
        args.extend([
            "--prefill-nodes", str(config['resources']['prefill_nodes']),
            "--decode-nodes", str(config['resources']['decode_nodes']),
            "--prefill-workers", str(config['resources']['prefill_workers']),
            "--decode-workers", str(config['resources']['decode_workers']),
        ])

    # Backend (for backward compatibility with GPU scripts)
    backend = config.get('backend', {})
    if 'gpu_type' in backend:
        args.extend(["--gpu-type", backend['gpu_type']])
    if 'script_variant' in backend:
        args.extend(["--script-variant", backend['script_variant']])

    # SGLang config path
    if sglang_config_path:
        args.extend(["--sglang-config-path", str(sglang_config_path)])

    # Backend environment variables
    if 'environment' in backend:
        env_json = json.dumps(backend['environment'])
        args.extend(["--backend-env", env_json])

    # Benchmark
    if 'benchmark' in config:
        benchmark_str = format_benchmark(config['benchmark'])
        args.extend(["--benchmark", benchmark_str])

    # Optional flags
    if config.get('use_init_location', False):
        args.append("--use-init-location")

    if config.get('enable_config_dump', True):
        # Enabled by default, only add flag if explicitly disabled
        if not config.get('enable_config_dump'):
            args.append("--disable-config-dump")

    return args


def format_benchmark(bench: dict) -> str:
    """Format benchmark dict to string (existing format)"""
    bench_type = bench.get('type', 'manual')

    if bench_type == 'sa-bench':
        # Handle both single value and list for concurrencies
        concurrencies = bench['concurrencies']
        if isinstance(concurrencies, list):
            concurrency_str = "x".join(str(c) for c in concurrencies)
        else:
            concurrency_str = str(concurrencies)

        return (
            f"type=sa-bench; "
            f"isl={bench['isl']}; "
            f"osl={bench['osl']}; "
            f"concurrencies={concurrency_str}; "
            f"req-rate={bench['req_rate']}"
        )
    elif bench_type == 'manual':
        return "type=manual"
    else:
        raise ValueError(f"Unknown benchmark type: {bench_type}")

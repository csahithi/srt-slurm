#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Submit jobs from YAML configs with SGLang config file support.

Usage:
    python submit_yaml.py config.yaml              # Single run
    python submit_yaml.py sweep.yaml --sweep       # Parameter sweep

Features:
    - Declarative YAML configs
    - SGLang config file generation (no more 50+ CLI flags!)
    - Environment variable management
    - Parameter sweeping (grid and list)
    - Backward compatible with existing submit_job_script.py
"""

import argparse
import itertools
import json
import logging
import os
import sys
import tempfile
import yaml
from pathlib import Path
from typing import Any

# Import existing submission logic
from submit_job_script import main as submit_job


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_yaml(path: Path) -> dict:
    """Load and validate YAML config"""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    # Basic validation
    required = ['name', 'slurm', 'resources', 'model', 'backend']
    for key in required:
        if key not in config:
            raise ValueError(f"Missing required key: {key}")

    logging.info(f"Loaded config: {config['name']}")
    return config


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

    # NEW: SGLang config path
    if sglang_config_path:
        args.extend(["--sglang-config-path", str(sglang_config_path)])

    # NEW: Backend environment variables
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


def submit_single(config_path: Path):
    """Submit a single job from YAML"""
    config = load_yaml(config_path)

    # Generate SGLang config if using sglang backend
    sglang_config_path = None
    if config.get('backend', {}).get('type') == 'sglang':
        sglang_config_path = generate_sglang_config_file(config)

    # Convert to args
    args = yaml_to_args(config, sglang_config_path)

    logging.info(f"Submitting job: {config['name']}")
    logging.info(f"Arguments: {' '.join(args)}")

    # Call existing submit_job_script
    submit_job(args)


def generate_sweep_configs(sweep_config: dict) -> list[tuple[dict, dict]]:
    """
    Generate all config combinations from sweep spec

    Returns:
        List of (config, params) tuples
    """
    if 'sweep' not in sweep_config:
        raise ValueError("Config must have 'sweep' section for sweep mode")

    sweep_spec = sweep_config['sweep']
    sweep_type = sweep_spec.get('type', 'grid')

    if sweep_type == 'grid':
        # Cartesian product of all parameters
        param_names = list(sweep_spec['parameters'].keys())
        param_values = list(sweep_spec['parameters'].values())

        configs = []
        for combination in itertools.product(*param_values):
            # Create parameter dict for this combination
            params = dict(zip(param_names, combination))

            # Expand templates in config
            expanded = expand_template(sweep_config, params)

            # Update name to include parameters
            param_str = "_".join(f"{k}{v}" for k, v in params.items())
            expanded['name'] = f"{sweep_config['name']}_{param_str}"

            configs.append((expanded, params))

        return configs

    elif sweep_type == 'list':
        # Explicit list of parameter combinations
        configs = []
        for param_set in sweep_spec['configs']:
            expanded = expand_template(sweep_config, param_set)
            param_str = "_".join(f"{k}{v}" for k, v in param_set.items())
            expanded['name'] = f"{sweep_config['name']}_{param_str}"
            configs.append((expanded, param_set))

        return configs

    else:
        raise ValueError(f"Unknown sweep type: {sweep_type}")


def submit_sweep(config_path: Path):
    """Submit parameter sweep"""
    sweep_config = load_yaml(config_path)

    # Generate all configs
    configs = generate_sweep_configs(sweep_config)

    logging.info(f"Generated {len(configs)} configurations for sweep")

    # TODO: Add job dependency tracking
    # For now, submit all jobs independently

    for i, (config, params) in enumerate(configs, 1):
        logging.info(f"\n[{i}/{len(configs)}] Submitting: {config['name']}")
        logging.info(f"  Parameters: {params}")

        # Generate SGLang config for this sweep iteration
        sglang_config_path = None
        if config.get('backend', {}).get('type') == 'sglang':
            sglang_config_path = generate_sglang_config_file(config, params)

        # Convert to args
        args = yaml_to_args(config, sglang_config_path)

        # Submit
        submit_job(args)


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Submit jobs from YAML configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single run
  python submit_yaml.py configs/my_run.yaml

  # Parameter sweep
  python submit_yaml.py configs/my_sweep.yaml --sweep
        """
    )
    parser.add_argument("config", type=Path, help="YAML config file")
    parser.add_argument("--sweep", action="store_true",
                       help="Treat as sweep config (multiple jobs)")

    args = parser.parse_args()

    if not args.config.exists():
        logging.error(f"Config file not found: {args.config}")
        sys.exit(1)

    try:
        if args.sweep:
            submit_sweep(args.config)
        else:
            submit_single(args.config)
    except Exception as e:
        logging.exception(f"Error submitting job: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

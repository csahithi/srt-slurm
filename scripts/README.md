# SLURM Runtime Scripts

This directory contains Jinja2 templates and runtime scripts for SLURM job execution.

## Contents

### Templates

- `job_script_template_disagg.j2` - Disaggregated prefill/decode mode
- `job_script_template_agg.j2` - Aggregated mode

These templates are used by the `srtctl` CLI to generate SLURM job scripts.

### Scripts Directory

Contains runtime scripts mounted into containers during job execution:

- `worker_setup.py` - Main worker launcher
- `slurm_utils.sh` - Utility functions for IP discovery
- `benchmark_utils.sh` - Benchmark helpers
- `check_server_health.py` - Health checks
- `monitor_gpu_utilization.sh` - GPU monitoring
- `nginx.conf.j2` - Nginx load balancer configuration
- `profile.sh` - Profiling utilities
- `sglang_bench_serving.sh` - SGLang benchmarking

### GPU-Specific Scripts

- `scripts/gb200-fp4/` - GB200 FP4 configurations
- `scripts/gb200-fp8/` - GB200 FP8 configurations
- `scripts/h100-fp8/` - H100 FP8 configurations

### Benchmark Scripts

- `scripts/sa-bench/` - SA-Bench workload
- `scripts/gpqa/` - GPQA evaluation
- `scripts/mmlu/` - MMLU evaluation

## Usage

**This directory is not meant to be used directly.** Instead, use the `srtctl` CLI:

```bash
# From repo root
uv run srtctl apply -f configs/gb200_fp4_max_tpt.yaml
```

See the main [README](../README.md) for documentation on the YAML-based submission flow.

## How It Works

1. User runs `srtctl apply -f config.yaml`
2. Backend generates SLURM script from templates in this directory
3. Scripts are mounted into containers at `/scripts/`
4. `worker_setup.py` launches SGLang workers with appropriate configs
5. Benchmarks run using scripts in `scripts/{benchmark}/`

## Custom Setup Scripts

You can provide custom setup scripts that run on each worker node before starting SGLang workers. Place your script in the `configs/` directory and pass it with `--setup-script`:

```bash
srtctl apply -f config.yaml --setup-script my-setup.sh
```

The script will be executed from `/configs/` inside the container. If no custom script is specified, the system looks for `setup-script.sh` by default. Setup scripts run before Dynamo installation from PyPI and worker startup, making them ideal for environment configuration and dependency installation.

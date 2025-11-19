# YAML Config Guide

This directory now supports YAML-based job configurations for cleaner, more maintainable benchmark submissions.

## Benefits

✅ **No more 50+ command line flags** - All SGLang flags in clean YAML
✅ **Environment variables version controlled** - No more scattered bash exports
✅ **Easy parameter sweeping** - Template substitution for sweep variables
✅ **Reproducible** - Full config saved with results
✅ **Type-safe** - YAML validation before job submission

## Quick Start

### Single Run

```bash
cd slurm_runner
python submit_yaml.py ../configs/example_sglang_disagg.yaml
```

### Parameter Sweep

```bash
python submit_yaml.py ../configs/example_mem_sweep.yaml --sweep
```

## Example Files

- **example_sglang_disagg.yaml** - Full SGLang disaggregated config with all flags
- **example_mem_sweep.yaml** - Simple sweep over mem_fraction_static
- **example_multi_dim_sweep.yaml** - Multi-dimensional sweep (72 jobs)

## Config Structure

```yaml
name: "my-benchmark"

slurm:
  account: "your-account"
  partition: "gpu"
  time_limit: "02:00:00"

resources:
  prefill_nodes: 1
  decode_nodes: 12
  prefill_workers: 1
  decode_workers: 1
  gpus_per_node: 4

model:
  path: "/models/your-model"
  container: "/containers/sglang.sqsh"

backend:
  type: "sglang"

  environment:
    PYTHONUNBUFFERED: "1"
    SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN: "1"

  sglang_config:
    shared:
      model_path: "/model/"
      kv_cache_dtype: "fp8_e4m3"

    prefill:
      mem_fraction_static: 0.84

    decode:
      mem_fraction_static: 0.85

benchmark:
  type: "sa-bench"
  isl: 1024
  osl: 1024
  concurrencies: [1024]
  req_rate: "inf"
```

## Sweeping Parameters

Use `{param_name}` as template variable:

```yaml
sglang_config:
  prefill:
    mem_fraction_static: "{mem_fraction}"

sweep:
  type: "grid"
  parameters:
    mem_fraction: [0.80, 0.84, 0.88]
```

This generates 3 jobs, one for each mem_fraction value.

## How It Works

1. `submit_yaml.py` generates SGLang config YAML from your config
2. Config is passed to workers via `--sglang-config-path`
3. `worker_setup.py` runs: `dynamo.sglang --config <path> --config-key prefill`
4. Coordination flags (--dist-init-addr, --node-rank) are appended automatically

## Backward Compatibility

Old command-line submissions still work. This is purely additive.

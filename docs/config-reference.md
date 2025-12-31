# Configuration Reference

Complete reference for job configuration YAML files.

## Overview

```yaml
name: "my-benchmark" # Required: job name

model: # Required: model settings
  path: "deepseek-r1"
  container: "latest"
  precision: "fp8"

resources: # Required: GPU allocation
  gpu_type: "gb200"
  prefill_nodes: 1
  decode_nodes: 2

slurm: # Optional: SLURM overrides
  time_limit: "02:00:00"

frontend: # Optional: router/frontend config
  type: dynamo

backend: # Optional: worker config
  type: sglang
  sglang_config:
    prefill: {}
    decode: {}

benchmark: # Optional: benchmark config
  type: "sa-bench"
  isl: 1024
  osl: 1024

dynamo: # Optional: dynamo version
  version: "0.7.0"

profiling: # Optional: nsys profiling
  enabled: false

setup_script: "my-setup.sh" # Optional: custom setup script
```

---

## model

Model and container configuration.

```yaml
model:
  path: "deepseek-r1" # Alias from srtslurm.yaml or full path
  container: "latest" # Container alias from srtslurm.yaml
  precision: "fp8" # fp8, fp4, bf16, etc.
```

| Field       | Type   | Description                                              |
| ----------- | ------ | -------------------------------------------------------- |
| `path`      | string | Model path alias (from `srtslurm.yaml`) or absolute path |
| `container` | string | Container alias (from `srtslurm.yaml`) or `.sqsh` path   |
| `precision` | string | Model precision (informational)                          |

---

## resources

GPU allocation and worker topology.

### Disaggregated Mode (prefill + decode)

```yaml
resources:
  gpu_type: "gb200"
  gpus_per_node: 4 # GPUs per node (default: from srtslurm.yaml)

  prefill_nodes: 2 # Nodes for prefill workers
  prefill_workers: 4 # Number of prefill workers

  decode_nodes: 4 # Nodes for decode workers
  decode_workers: 8 # Number of decode workers
```

### Aggregated Mode (single worker type)

```yaml
resources:
  gpu_type: "h100"
  gpus_per_node: 8
  agg_nodes: 2 # Nodes for aggregated workers
  agg_workers: 4 # Number of aggregated workers
```

| Field             | Type   | Default            | Description                  |
| ----------------- | ------ | ------------------ | ---------------------------- |
| `gpu_type`        | string | -                  | GPU type identifier          |
| `gpus_per_node`   | int    | from srtslurm.yaml | GPUs per node                |
| `prefill_nodes`   | int    | 0                  | Nodes dedicated to prefill   |
| `decode_nodes`    | int    | 0                  | Nodes dedicated to decode    |
| `prefill_workers` | int    | 1                  | Number of prefill workers    |
| `decode_workers`  | int    | 1                  | Number of decode workers     |
| `agg_nodes`       | int    | 0                  | Nodes for aggregated mode    |
| `agg_workers`     | int    | 1                  | Number of aggregated workers |

**Note**: Set `decode_nodes: 0` to have decode workers share nodes with prefill workers.

---

## slurm

SLURM job settings.

```yaml
slurm:
  time_limit: "04:00:00" # Job time limit
  account: "my-account" # SLURM account (overrides srtslurm.yaml)
  partition: "batch" # SLURM partition (overrides srtslurm.yaml)
```

| Field        | Type   | Default            | Description               |
| ------------ | ------ | ------------------ | ------------------------- |
| `time_limit` | string | from srtslurm.yaml | Job time limit (HH:MM:SS) |
| `account`    | string | from srtslurm.yaml | SLURM account             |
| `partition`  | string | from srtslurm.yaml | SLURM partition           |

---

## frontend

Frontend/router configuration.

```yaml
frontend:
  # Frontend type: "dynamo" (default) or "sglang"
  type: dynamo

  # Scaling
  enable_multiple_frontends: true # Enable nginx + multiple routers
  num_additional_frontends: 9 # Additional routers (total = 1 + this)

  # CLI args passed to the frontend/router
  args:
    router-mode: "kv" # dynamo: router-mode
    policy: "cache_aware" # sglang: policy
    no-kv-events: true # boolean flags

  # Environment variables for frontend processes
  env:
    MY_VAR: "value"
```

| Field                       | Type | Default | Description                         |
| --------------------------- | ---- | ------- | ----------------------------------- |
| `type`                      | str  | dynamo  | Frontend type: "dynamo" or "sglang" |
| `enable_multiple_frontends` | bool | true    | Scale with nginx + multiple routers |
| `num_additional_frontends`  | int  | 9       | Additional routers beyond master    |
| `args`                      | dict | null    | CLI args for the frontend           |
| `env`                       | dict | null    | Env vars for frontend processes     |

See [SGLang Router](sglang-router.md) for detailed architecture.

---

## backend

Worker configuration and SGLang settings.

```yaml
backend:
  type: sglang # Backend type (currently only sglang)

  # Per-mode environment variables
  prefill_environment:
    TORCH_DISTRIBUTED_DEFAULT_TIMEOUT: "1800"
  decode_environment:
    TORCH_DISTRIBUTED_DEFAULT_TIMEOUT: "1800"
  aggregated_environment: {}

  # SGLang CLI config per mode
  sglang_config:
    prefill:
      tensor-parallel-size: 4
      mem-fraction-static: 0.84
      kv-cache-dtype: "fp8_e4m3"
      disaggregation-mode: "prefill"
      # ... any sglang CLI flag
    decode:
      tensor-parallel-size: 8
      mem-fraction-static: 0.83
      data-parallel-size: 8
      enable-dp-attention: true
    aggregated:
      # ... for aggregated mode

  # KV events (for kv-aware routing)
  kv_events_config:
    prefill: true # Enable for prefill workers
    decode: true # Enable for decode workers
```

### sglang_config

Any SGLang CLI flag can be specified (use kebab-case or snake_case):

| Common Flags                      | Description                |
| --------------------------------- | -------------------------- |
| `tensor-parallel-size`            | TP degree                  |
| `data-parallel-size`              | DP degree                  |
| `expert-parallel-size`            | EP degree (MoE models)     |
| `mem-fraction-static`             | GPU memory fraction        |
| `kv-cache-dtype`                  | KV cache precision         |
| `context-length`                  | Max context length         |
| `chunked-prefill-size`            | Chunked prefill batch size |
| `enable-dp-attention`             | Enable DP attention        |
| `disaggregation-mode`             | "prefill" or "decode"      |
| `disaggregation-transfer-backend` | "nixl" or other            |

### kv_events_config

Enables `--kv-events-config` for workers with auto-allocated ZMQ ports.

```yaml
# Enable with defaults
kv_events_config: true         # prefill+decode with publisher=zmq, topic=kv-events

# Per-mode control
kv_events_config:
  prefill: true
  decode: true
  # agg: omitted = disabled

# Custom settings
kv_events_config:
  prefill:
    publisher: "zmq"
    topic: "prefill-events"
  decode:
    topic: "decode-events"     # publisher defaults to "zmq"
```

Each worker leader gets a globally unique port starting at 5550:

| Worker    | Port |
| --------- | ---- |
| prefill_0 | 5550 |
| prefill_1 | 5551 |
| decode_0  | 5552 |
| decode_1  | 5553 |

---

## benchmark

Benchmark configuration.

### sa-bench (Serving Accuracy)

```yaml
benchmark:
  type: "sa-bench"
  isl: 1024 # Input sequence length
  osl: 1024 # Output sequence length
  concurrencies: [256, 512] # Concurrency levels to test
  req_rate: "inf" # Request rate (or number)
```

### mooncake-router

```yaml
benchmark:
  type: "mooncake-router"
  mooncake_workload: "conversation"
  ttft_threshold_ms: 2000
  itl_threshold_ms: 25
```

| Field           | Type        | Description                                         |
| --------------- | ----------- | --------------------------------------------------- |
| `type`          | string      | Benchmark type: `sa-bench`, `mooncake-router`, etc. |
| `isl`           | int         | Input sequence length                               |
| `osl`           | int         | Output sequence length                              |
| `concurrencies` | list/string | Concurrency levels (list or "NxM" format)           |
| `req_rate`      | string      | Request rate                                        |

---

## dynamo

Dynamo version configuration.

```yaml
dynamo:
  version: "0.7.0" # Install from PyPI
  # OR
  hash: "abc123" # Install from git commit
  # OR
  top_of_tree: true # Install from main branch
```

| Field         | Type   | Default | Description                      |
| ------------- | ------ | ------- | -------------------------------- |
| `version`     | string | "0.7.0" | PyPI version                     |
| `hash`        | string | null    | Git commit hash (source install) |
| `top_of_tree` | bool   | false   | Install from main branch         |

**Note**: `hash` and `top_of_tree` are mutually exclusive.

---

## profiling

Enable nsys profiling for workers.

```yaml
profiling:
  enabled: true
  target_workers:
    - "prefill_0" # Profile first prefill worker
    - "decode_0" # Profile first decode worker
  nsys_args:
    trace: "cuda,nvtx"
    duration: 120
```

| Field            | Type | Default | Description               |
| ---------------- | ---- | ------- | ------------------------- |
| `enabled`        | bool | false   | Enable nsys profiling     |
| `target_workers` | list | []      | Workers to profile        |
| `nsys_args`      | dict | {}      | Additional nsys arguments |

See [Profiling](profiling.md) for details.

---

## Other Options

### setup_script

Run a custom script before worker startup:

```yaml
setup_script: "install-sglang-main.sh"
```

Script must be in `configs/` directory. See [Installation](installation.md#custom-setup-scripts).

### environment

Global environment variables for all workers:

```yaml
environment:
  MY_VAR: "value"
  CUDA_LAUNCH_BLOCKING: "1"
```

### extra_mount

Additional container mounts:

```yaml
extra_mount:
  - "/local/path:/container/path"
  - "/data:/data:ro"
```

### sbatch_directives

Additional SLURM sbatch directives:

```yaml
sbatch_directives:
  mail-user: "user@example.com"
  mail-type: "END,FAIL"
```

### srun_options

Additional srun options:

```yaml
srun_options:
  cpu-bind: "none"
```

---

## Complete Example

```yaml
name: "deepseek-r1-benchmark"

model:
  path: "deepseek-r1"
  container: "0.5.6"
  precision: "fp8"

resources:
  gpu_type: "gb200"
  gpus_per_node: 4
  prefill_nodes: 2
  prefill_workers: 4
  decode_nodes: 4
  decode_workers: 8

slurm:
  time_limit: "04:00:00"

frontend:
  type: dynamo
  enable_multiple_frontends: true
  args:
    router-mode: "kv"

backend:
  type: sglang

  kv_events_config:
    prefill: true

  prefill_environment:
    TORCH_DISTRIBUTED_DEFAULT_TIMEOUT: "1800"
  decode_environment:
    TORCH_DISTRIBUTED_DEFAULT_TIMEOUT: "1800"

  sglang_config:
    prefill:
      tensor-parallel-size: 4
      mem-fraction-static: 0.84
      kv-cache-dtype: "fp8_e4m3"
    decode:
      tensor-parallel-size: 8
      mem-fraction-static: 0.83
      data-parallel-size: 8

benchmark:
  type: "sa-bench"
  isl: 1024
  osl: 1024
  concurrencies: [128, 256, 512]

dynamo:
  version: "0.7.0"
```

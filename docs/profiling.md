# Profiling

srtctl supports two profiling backends for performance analysis: **Torch Profiler** and **NVIDIA Nsight Systems (nsys)**. Profiling helps identify bottlenecks in prefill and decode operations.

## Quick Start

Add a `profiling` section to your job YAML:

```yaml
profiling:
  type: "torch" # or "nsys"
  prefill:
    isl: 1024
    osl: 128
    concurrency: 24
  decode:
    isl: 1024
    osl: 128
    concurrency: 256
```

## Profiling Modes

| Mode    | Description                                                      | Output                                         |
| ------- | ---------------------------------------------------------------- | ---------------------------------------------- |
| `none`  | Default. No profiling, uses `dynamo.sglang` for serving          | -                                              |
| `torch` | PyTorch Profiler. Good for Python-level and CUDA kernel analysis | `/logs/profiles/{mode}/` (Chrome trace format) |
| `nsys`  | NVIDIA Nsight Systems. Low-overhead GPU profiling                | `/logs/profiles/{mode}_{rank}.nsys-rep`        |

## Configuration Options

### Top-level `profiling` section

```yaml
profiling:
  type: "torch" # Required: "none", "torch", or "nsys"

  prefill: # Optional: prefill-specific parameters
    isl: 1024 # Input sequence length
    osl: 128 # Output sequence length
    concurrency: 24 # Batch size for profiling workload
    start_step: 0 # Step to start profiling
    stop_step: 50 # Step to stop profiling

  decode: # Optional: decode-specific parameters
    isl: 1024
    osl: 128
    concurrency: 256
    start_step: 0
    stop_step: 50
```

### Phase Parameters

| Parameter     | Description                                   | Default  |
| ------------- | --------------------------------------------- | -------- |
| `isl`         | Input sequence length for profiling requests  | Required |
| `osl`         | Output sequence length for profiling requests | Required |
| `concurrency` | Number of concurrent requests (batch size)    | Required |
| `start_step`  | Step number to begin profiling                | `0`      |
| `stop_step`   | Step number to end profiling                  | `50`     |

## Constraints

Profiling has specific requirements:

1. **Single worker only**: Profiling requires exactly 1 prefill worker and 1 decode worker (or 1 aggregated worker)

   ```yaml
   resources:
     prefill_workers: 1 # Must be 1
     decode_workers: 1 # Must be 1
   ```

2. **No benchmarking**: Profiling and benchmarking are mutually exclusive

   ```yaml
   benchmark:
     type: "manual" # Required when profiling
   ```

3. **Automatic config dump disabled**: When profiling is enabled, `enable_config_dump` is automatically set to `false`

## How It Works

### Normal Mode (`type: none`)

- Uses `dynamo.sglang` module for serving
- Standard disaggregated inference path

### Profiling Mode (`type: torch` or `nsys`)

- Uses `sglang.launch_server` module instead
- The `--disaggregation-mode` flag is automatically skipped (not supported by launch_server)
- Profiling script (`/scripts/profiling/profile.sh`) runs on leader nodes
- Sends requests via `sglang.bench_serving` to generate profiling workload

### nsys-specific behavior

When using `nsys`, workers are wrapped with:

```bash
nsys profile -t cuda,nvtx --cuda-graph-trace=node \
  -c cudaProfilerApi --capture-range-end stop \
  -o /logs/profiles/{mode}_{rank} \
  python3 -m sglang.launch_server ...
```

## Example Configurations

### Torch Profiler (Recommended for Python analysis)

```yaml
name: "profiling-torch"

model:
  path: "dsfp8"
  container: "0.5.5"
  precision: "fp8"

resources:
  gpu_type: "gb200"
  prefill_nodes: 1
  decode_nodes: 1
  prefill_workers: 1
  decode_workers: 1
  gpus_per_node: 4

profiling:
  type: "torch"
  prefill:
    isl: 1024
    osl: 128
    concurrency: 24
    start_step: 0
    stop_step: 50
  decode:
    isl: 1024
    osl: 128
    concurrency: 256
    start_step: 0
    stop_step: 50

benchmark:
  type: "manual"

backend:
  sglang_config:
    prefill:
      model-path: "/model/"
      # ... other flags
    decode:
      model-path: "/model/"
      # ... other flags
```

### Nsight Systems (Recommended for GPU kernel analysis)

```yaml
profiling:
  type: "nsys"
  prefill:
    isl: 2048
    osl: 64
    concurrency: 16
    start_step: 10
    stop_step: 30
  decode:
    isl: 2048
    osl: 64
    concurrency: 512
    start_step: 10
    stop_step: 30
```

## Output Files

After profiling completes, find results in the job's log directory:

```
logs/{job_id}_{workers}_{timestamp}/
├── profile_prefill.out     # Prefill profiling script output
├── profile_decode.out      # Decode profiling script output
└── profiles/
    ├── prefill/            # Torch profiler traces (if type: torch)
    │   └── *.json
    ├── decode/
    │   └── *.json
    ├── prefill_0.nsys-rep  # Nsys reports (if type: nsys)
    └── decode_0.nsys-rep
```

### Viewing Results

**Torch Profiler traces:**

- Open in Chrome: `chrome://tracing`
- Or use TensorBoard: `tensorboard --logdir=logs/.../profiles/`

**Nsight Systems reports:**

- Open with NVIDIA Nsight Systems GUI
- Or CLI: `nsys stats logs/.../profiles/decode_0.nsys-rep`

## Troubleshooting

### "Profiling mode requires single worker only"

Reduce your worker counts to 1:

```yaml
resources:
  prefill_workers: 1
  decode_workers: 1
```

### "Cannot enable profiling with benchmark type"

Set benchmark to manual:

```yaml
benchmark:
  type: "manual"
```

### Empty profile output

Ensure `isl`, `osl`, and `concurrency` are set - they're required for the profiling workload.

### Profile too short/long

Adjust `start_step` and `stop_step` to capture the desired range. A typical profiling run uses 30-100 steps.

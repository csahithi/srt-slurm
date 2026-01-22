# srtctl - Python-first SLURM Orchestration

This package provides Python-first orchestration for LLM inference benchmarks
on SLURM clusters, replacing the previous Jinja/bash-heavy approach.

## Architecture

```
srtctl/
├── __init__.py              # Package exports
├── cli/
│   ├── submit.py            # srtctl apply - job submission
│   ├── do_sweep.py          # srtctl-sweep - main orchestrator
│   ├── setup_head.py        # Head node infrastructure (NATS/etcd)
│   └── mixins/
│       ├── worker_stage.py  # Worker startup mixin
│       ├── frontend_stage.py# Frontend/NGINX startup mixin
│       ├── benchmark_stage.py# Benchmark execution mixin
│       └── rollup_stage.py  # Experiment data consolidation
├── core/
│   ├── config.py            # Config loading and srtslurm.yaml resolution
│   ├── runtime.py           # RuntimeContext - single source of truth
│   ├── topology.py          # Endpoint/Process allocation for workers
│   ├── processes.py         # ProcessRegistry - lifecycle management
│   ├── slurm.py             # SLURM srun launching and node resolution
│   ├── health.py            # Health checks (HTTP polling, worker readiness)
│   ├── schema.py            # Frozen dataclass schemas
│   ├── sweep.py             # Sweep parameter handling
│   └── ip_utils/            # Bash-based IP resolution utilities
│       ├── __init__.py      # Python wrappers for bash functions
│       └── get_node_ip.sh   # IP detection bash functions
├── backends/
│   ├── base.py              # BackendProtocol interface
│   ├── sglang.py            # SGLang implementation
│   └── trtllm.py            # TensorRT-LLM implementation
├── benchmarks/
│   ├── base.py              # BenchmarkRunner ABC
│   ├── sa_bench.py          # Serving benchmark
│   ├── router.py            # Router benchmark
│   └── ...                  # Other benchmark types
└── templates/               # Jinja2 templates for sbatch scripts
```

## Usage

```bash
srtctl apply -f config.yaml
```

## Key Concepts

### RuntimeContext

Single source of truth for all computed paths and values. Replaces bash
variables scattered throughout Jinja templates.

```python
runtime = RuntimeContext.from_config(config, job_id)
print(runtime.log_dir)       # Computed once
print(runtime.model_path)    # Resolved from config
print(runtime.head_node_ip)  # From SLURM
```

### Endpoints and Processes

Typed Python replaces bash array math:

```python
# Old (Jinja/bash):
# for i in $(seq 0 $((PREFILL_WORKERS - 1))); do
#     leader_idx=$((WORKER_NODE_OFFSET + i * PREFILL_NODES_PER_WORKER))
# done

# New (Python):
endpoints = allocate_endpoints(
    num_prefill=2, num_decode=4, num_agg=0,
    gpus_per_prefill=8, gpus_per_decode=4, gpus_per_agg=8,
    gpus_per_node=8, available_nodes=nodes
)
for endpoint in endpoints:
    print(f"{endpoint.mode} worker {endpoint.index} on {endpoint.nodes}")
```

### ProcessRegistry

Manages process lifecycle with health monitoring:

```python
registry = ProcessRegistry(job_id)
registry.add_process(worker_proc)

# Background thread monitors for failures
if registry.check_failures():
    registry.cleanup()  # Graceful shutdown
```

### Health Checks

HTTP-based health checking for different frontends:

```python
from srtctl.core.health import wait_for_model

# Wait for all workers to register
wait_for_model(
    host=head_ip, port=8000,
    n_prefill=2, n_decode=4,
    frontend_type="sglang",  # or "dynamo"
    timeout=300,
)
```

For aggregated mode, pass `n_prefill=0, n_decode=num_agg`.

### BackendProtocol

Interface for different serving frameworks:

```python
class BackendProtocol(Protocol):
    @property
    def type(self) -> BackendType: ...
    def build_worker_command(self, process, runtime) -> list[str]: ...
```

### Multiple Workers Per Node

The allocator automatically handles placing multiple workers on a single node:

```yaml
resources:
  gpus_per_node: 8
  decode_workers: 2
  gpus_per_decode: 4 # 2 workers × 4 GPUs = 8 GPUs = 1 node
```

`CUDA_VISIBLE_DEVICES` is automatically set per worker (e.g., `0,1,2,3` and `4,5,6,7`).

### Rollup Stage

After benchmark completion, the rollup stage consolidates all experiment data into
a single `rollup.json` file for easy analysis:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ROLLUP STAGE PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT FILES                          OUTPUT                                │
│  ───────────                          ──────                                │
│  • benchmark results (*.json)    ──┐                                        │
│  • worker logs (*.out/*.err)     ──┼──►  rollup.json                        │
│  • config.yaml                   ──┤     • benchmark results                │
│  • engine configs (*.yaml)       ──┘     • node metrics                     │
│                                          • environment config               │
│                                          • summary statistics               │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Data collected:**

| Component | Source | Output |
|-----------|--------|--------|
| Benchmark Results | `sa-bench_*/results_*.json` | `RollupResult[]` with TPS, latencies |
| Node Metrics | `worker-*.out` logs | `NodesSummary` with batches, memory, throughput |
| Environment Config | `config.yaml`, `trtllm_config_*.yaml` | `EnvironmentConfig` with env vars, engine settings |
| Launch Commands | `benchmark.out`, worker logs | Parsed command parameters |

**Modular Parser System:**

The rollup uses pluggable parsers from `analysis.srtlog.parsers`:

```python
# Benchmark parsers (parse result JSON files)
from analysis.srtlog.parsers import get_benchmark_parser
parser = get_benchmark_parser("sa-bench")  # or "mooncake-router"
results = parser.parse_result_directory(log_dir)

# Node parsers (parse worker log files)  
from analysis.srtlog.parsers import get_node_parser
parser = get_node_parser("trtllm")  # or "sglang"
nodes = parser.parse_logs(log_dir)
```

**Example rollup.json structure:**

```json
{
  "job_id": "12345",
  "job_name": "disagg-benchmark",
  "model_path": "/model/llama-70b",
  "backend_type": "trtllm",
  
  "results": [
    {"concurrency": 16, "output_tps": 2500.0, "mean_ttft_ms": 45.2, ...},
    {"concurrency": 32, "output_tps": 4000.0, "mean_ttft_ms": 52.1, ...}
  ],
  
  "nodes_summary": {
    "total_prefill_nodes": 1,
    "total_decode_nodes": 7,
    "avg_decode_gen_throughput": 533.1,
    "total_kv_cache_gb": 325.0,
    "nodes": [
      {
        "node_name": "worker-0",
        "worker_type": "prefill",
        "total_batches": 1523,
        "avg_input_throughput": 21565.5,
        "mem_usage_gb": 91.46,
        "kv_cache_gb": 41.19
      }
    ]
  },
  
  "environment_config": {
    "prefill_environment": {"UCX_TLS": "rc,dc,ud,...", "TRTLLM_ENABLE_PDL": "1"},
    "decode_environment": {"UCX_TLS": "rc,dc,ud,..."},
    "prefill_engine_config": {"tensor_parallel_size": 8, "max_batch_size": 2},
    "decode_engine_config": {"tensor_parallel_size": 8, "max_batch_size": 32}
  },
  
  "max_output_tps": 4000.0,
  "min_mean_ttft_ms": 45.2
}
```

## Files Overview

| File | Purpose |
| ---- | ------- |
| `core/config.py` | YAML loading, srtslurm.yaml resolution |
| `core/runtime.py` | Computed paths/values (RuntimeContext) |
| `core/topology.py` | Worker topology and GPU allocation |
| `core/processes.py` | Process lifecycle management |
| `core/slurm.py` | SLURM srun launching, node IP resolution |
| `core/health.py` | Health checks, worker readiness polling |
| `core/ip_utils/` | Bash-based IP detection utilities |
| `cli/do_sweep.py` | Main orchestrator (runs on head node) |
| `cli/mixins/rollup_stage.py` | Experiment data consolidation to rollup.json |
| `backends/sglang.py` | SGLang backend implementation |
| `backends/trtllm.py` | TensorRT-LLM backend implementation |
| `benchmarks/base.py` | BenchmarkRunner ABC |

### Related Analysis Modules

| File | Purpose |
| ---- | ------- |
| `analysis/srtlog/parsers/__init__.py` | Parser registry and protocols |
| `analysis/srtlog/parsers/benchmark/sa_bench.py` | SA-Bench result parser |
| `analysis/srtlog/parsers/benchmark/mooncake_router.py` | Mooncake router result parser |
| `analysis/srtlog/parsers/nodes/sglang.py` | SGLang worker log parser |
| `analysis/srtlog/parsers/nodes/trtllm.py` | TRTLLM worker log parser |
| `analysis/srtlog/models.py` | Data models (NodeMetrics, BatchMetrics, etc.) |

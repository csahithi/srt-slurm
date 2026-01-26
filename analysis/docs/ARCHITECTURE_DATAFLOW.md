# SRT-SLURM Log Analysis Architecture - Dataflow Diagram

## Overview
This document describes the data flow through the log analysis system, from raw log files to structured data models.

---

## 1. Entry Point: RunLoader

```
┌─────────────────────────────────────────────────────────────────────┐
│                           RunLoader                                  │
│  Entry point for loading and analyzing benchmark run data           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ├──► discover_runs()
                                    ├──► load_single(job_id)
                                    └──► load_node_metrics_for_run()
                                    
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
        ┌─────────────────────┐         ┌────────────────────┐
        │  Metadata Discovery │         │  Results Parsing   │
        └─────────────────────┘         └────────────────────┘
```

---

## 2. Metadata Discovery Flow

```
                        ┌─────────────────────────────────┐
                        │    Source Files (per run)       │
                        │                                 │
                        │  📁 {job_id}/metadata.json      │
                        │  📁 {job_id}/config.yaml        │
                        │  📁 {job_id}/*.json             │
                        └─────────────────────────────────┘
                                      │
                                      │ read by
                                      ▼
                        ┌─────────────────────────────────┐
                        │   RunLoader._load_metadata()    │
                        └─────────────────────────────────┘
                                      │
                                      │ creates
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                            RunMetadata                                     │
│  Fields:                                   Source File:                    │
│  • job_id                                  📁 metadata.json                │
│  • job_name                                📁 metadata.json                │
│  • run_date                                📁 metadata.json                │
│  • mode (monolithic/disaggregated)         📁 metadata.json                │
│  • prefill_nodes, decode_nodes             📁 metadata.json                │
│  • prefill_workers, decode_workers         📁 metadata.json                │
│  • model: ModelConfig                      📁 metadata.json                │
│    - path, tensor_parallel, ...                                            │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Profiler/Benchmark Results Flow

```
                        ┌─────────────────────────────────────────┐
                        │        Profiler Type Detection          │
                        │                                         │
                        │  📁 logs/benchmark.out                  │
                        │    - Search for "SA-Bench Config"       │
                        │    - Search for "aiperf" commands       │
                        └─────────────────────────────────────────┘
                                         │
                                         │ determines
                                         ▼
                        ┌─────────────────────────────────────────┐
                        │       ProfilerMetadata                  │
                        │  Fields:              Source:           │
                        │  • profiler_type      benchmark.out     │
                        │  • isl                benchmark.out     │
                        │  • osl                benchmark.out     │
                        │  • concurrencies      benchmark.out     │
                        └─────────────────────────────────────────┘
                                         │
                                         │ used to find
                                         ▼
        ┌────────────────────────────────────────────────────────────────┐
        │              BenchmarkParser.find_result_directory()           │
        │                                                                │
        │  SA-Bench:                    Mooncake-Router:                │
        │  📁 sa-bench_isl_*_osl_*/     📁 logs/artifacts/*/            │
        │     result_*.json (PRIMARY)      profile_export_aiperf.json   │
        │     benchmark.out (FALLBACK)     (PRIMARY)                    │
        │                               📁 logs/benchmark.out           │
        │                                  (FALLBACK)                   │
        └────────────────────────────────────────────────────────────────┘
                                         │
                                         │ parse_result_directory()
                                         │ ⚠️ JSON files are PRIMARY source of truth
                                         │    .out files are FALLBACK only
                                         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          ProfilerResults                                  │
│  Fields:                               Source Files (Priority Order):     │
│  • output_tps: list[float]             1️⃣ 📁 result_*.json (SA-Bench)    │
│  • request_throughput: list[float]        📁 profile_export_aiperf.json   │
│  • concurrency_values: list[int]             (Mooncake-Router)            │
│  • mean_ttft_ms: list[float]           2️⃣ 📁 logs/benchmark.out (fallback)│
│  • mean_itl_ms: list[float]                                               │
│  • mean_e2el_ms: list[float]            One entry per concurrency level   │
│  • p99_ttft_ms, median_ttft_ms, ...                                       │
│  • total_input_tokens: list[int]        JSON = Source of Truth ✨         │
│  • total_output_tokens: list[int]       .out = Fallback only ⚠️           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Benchmark Launch Command Flow

```
                        ┌─────────────────────────────────┐
                        │    Source File                  │
                        │  📁 logs/benchmark.out          │
                        │    - Command line arguments     │
                        │    - SA-Bench Config: header    │
                        │    - aiperf profile commands    │
                        └─────────────────────────────────┘
                                      │
                                      │ parse_launch_command()
                                      ▼
                        ┌─────────────────────────────────┐
                        │   BenchmarkParser               │
                        │   (SA-Bench or Mooncake)        │
                        └─────────────────────────────────┘
                                      │
                                      │ creates
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                      BenchmarkLaunchCommand                                │
│  Fields:                                   Source:                         │
│  • benchmark_type                          📁 logs/benchmark.out           │
│  • raw_command                             📁 logs/benchmark.out           │
│  • extra_args: dict                        📁 logs/benchmark.out           │
│    - base_url, model, input_len,                                           │
│      output_len, max_concurrency, ...                                      │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Node Metrics Flow

```
                        ┌─────────────────────────────────────────────────┐
                        │          Source Files (per node/worker)         │
                        │                                                 │
                        │  📁 logs/{node}_{worker_type}_{worker_id}.out  │
                        │     Examples:                                   │
                        │     - worker-3_decode_w0.out                   │
                        │     - eos0219_prefill_w1.out                   │
                        │                                                 │
                        │  Content:                                       │
                        │  • Batch metrics lines                         │
                        │  • Memory snapshot lines                       │
                        │  • TP/DP/EP configuration                      │
                        │  • Launch command                              │
                        └─────────────────────────────────────────────────┘
                                         │
                                         │ detect backend type
                                         ▼
                        ┌─────────────────────────────────┐
                        │   NodeAnalyzer                  │
                        │   _detect_backend_type()        │
                        │   • Checks config.yaml          │
                        │   • Checks log patterns         │
                        └─────────────────────────────────┘
                                         │
                                         │ get_node_parser()
                                         ▼
        ┌────────────────────────────────────────────────────────┐
        │              NodeParser (SGLang or TRT-LLM)            │
        │                                                        │
        │  parse_single_log() - parses one worker's log file    │
        └────────────────────────────────────────────────────────┘
                                         │
                                         │ creates
                                         ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                            NodeMetadata                                    │
│  Fields:                               Source:                             │
│  • node_name                           📁 *_{type}_{id}.out (filename)     │
│  • worker_type (prefill/decode/agg)    📁 *_{type}_{id}.out (filename)     │
│  • worker_id (w0, w1, ...)             📁 *_{type}_{id}.out (filename)     │
└───────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                            BatchMetrics                                    │
│  Fields:                               Source:                             │
│  • timestamp                           📁 *.out log lines                  │
│  • dp, tp, ep                          📁 *.out log lines                  │
│  • batch_type (prefill/decode)         📁 *.out log lines                  │
│  • new_seq, new_token, cached_token    📁 *.out log lines                  │
│  • token_usage                         📁 *.out log lines                  │
│  • running_req, queue_req              📁 *.out log lines                  │
│  • num_tokens                          📁 *.out log lines                  │
│  • input_throughput, gen_throughput    📁 *.out log lines                  │
│                                                                            │
│  Example log line (SGLang):                                               │
│  2024-12-30 08:10:15 DP0.TP0.EP0 [BATCH] prefill #new-seq: 2 ...         │
│                                                                            │
│  Example log line (TRT-LLM):                                              │
│  [TensorRT-LLM][INFO] [ITERATION] tokens=1024 new_tokens=128 ...         │
└───────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                            MemoryMetrics                                   │
│  Fields:                               Source:                             │
│  • timestamp                           📁 *.out log lines                  │
│  • dp, tp, ep                          📁 *.out log lines                  │
│  • avail_mem_gb                        📁 *.out log lines                  │
│  • mem_usage_gb                        📁 *.out log lines                  │
│  • kv_cache_gb                         📁 *.out log lines                  │
│  • kv_tokens                           📁 *.out log lines                  │
│                                                                            │
│  Example log line (SGLang):                                               │
│  2024-12-30 08:10:15 DP0.TP0.EP0 #running-req: 10, avail_mem=45.2GB      │
│                                                                            │
│  Example log line (TRT-LLM):                                              │
│  [TensorRT-LLM][INFO] Memory Stats: free=48.5GB, kv_cache=12.3GB         │
└───────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                            NodeMetrics                                     │
│  Fields:                               Source:                             │
│  • metadata: NodeMetadata              (see above)                         │
│  • batches: list[BatchMetrics]         📁 *.out log lines                  │
│  • memory_snapshots: list[MemoryMetrics] 📁 *.out log lines                │
│  • config: dict                        📁 *.out log lines                  │
│    - tp_size, dp_size, ep_size         (parsed from DP0.TP2.EP1 tags)     │
│  • run_id                              (from metadata)                     │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Node Configuration Flow

```
                        ┌─────────────────────────────────────────────────┐
                        │          Source Files (per node)                │
                        │                                                 │
                        │  📁 logs/*_{type}_{id}.out - launch command    │
                        │  📁 logs/*_config.json - node config           │
                        │  📁 logs/config.yaml - environment vars        │
                        └─────────────────────────────────────────────────┘
                                         │
                                         │ parsed by
                                         ▼
                        ┌─────────────────────────────────┐
                        │   NodeAnalyzer                  │
                        │   _populate_config_from_files() │
                        └─────────────────────────────────┘
                                         │
                                         │ creates
                                         ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                      NodeLaunchCommand                                     │
│  Fields:                               Source:                             │
│  • backend_type (sglang/trtllm)        📁 *_{type}_{id}.out               │
│  • worker_type (prefill/decode)        📁 *_{type}_{id}.out               │
│  • raw_command                         📁 *_{type}_{id}.out               │
│  • extra_args: dict                    📁 *_{type}_{id}.out               │
│    - model_path, served_model_name,                                        │
│      disaggregation_mode, tp_size,                                         │
│      pp_size, max_num_seqs, ...                                            │
│                                                                            │
│  Example (TRT-LLM):                                                        │
│  python3 -m dynamo.trtllm --model-path /model --disaggregation-mode       │
│    decode --extra-engine-args /logs/trtllm_config_decode.yaml             │
│                                                                            │
│  Example (SGLang):                                                         │
│  python -m sglang.launch_server --model-path /model --disagg-mode prefill │
│    --tp-size 2 --dp-size 1                                                 │
└───────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                          NodeConfig (TypedDict)                            │
│  Fields:                               Source:                             │
│  • launch_command: NodeLaunchCommand   📁 *_{type}_{id}.out               │
│  • environment: dict[str, str]         📁 config.yaml                     │
│    - NCCL settings, CUDA settings,                                         │
│      model paths, etc.                                                     │
│  • gpu_info: dict (optional)           📁 *_config.json                   │
└───────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                            NodeInfo                                        │
│  Top-level container combining metrics and configuration                  │
│                                                                            │
│  Fields:                                                                   │
│  • metrics: NodeMetrics                (performance data)                  │
│  • node_config: NodeConfig             (configuration)                    │
│                                                                            │
│  Convenience properties delegate to nested fields:                         │
│  • node_name → metrics.metadata.node_name                                 │
│  • worker_type → metrics.metadata.worker_type                             │
│  • launch_command → node_config["launch_command"]                         │
│  • environment → node_config["environment"]                               │
│  • batches → metrics.batches                                              │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Complete Data Model Hierarchy

```
BenchmarkRun (top-level container for entire run)
│
├─ metadata: RunMetadata
│  └─ Source: 📁 metadata.json, config.yaml
│
├─ profiler_metadata: ProfilerMetadata
│  └─ Source: 📁 logs/benchmark.out
│
├─ profiler: ProfilerResults
│  └─ Source: 📁 sa-bench_isl_*_osl_*/result_*.json
│              📁 logs/artifacts/*/profile_export_aiperf.json
│
├─ benchmark_launch_command: BenchmarkLaunchCommand
│  └─ Source: 📁 logs/benchmark.out
│
└─ nodes: list[NodeInfo]
   └─ Each NodeInfo contains:
      │
      ├─ metrics: NodeMetrics
      │  ├─ metadata: NodeMetadata
      │  │  └─ Source: 📁 logs/*_{type}_{id}.out (filename)
      │  ├─ batches: list[BatchMetrics]
      │  │  └─ Source: 📁 logs/*_{type}_{id}.out (log lines)
      │  ├─ memory_snapshots: list[MemoryMetrics]
      │  │  └─ Source: 📁 logs/*_{type}_{id}.out (log lines)
      │  └─ config: dict
      │     └─ Source: 📁 logs/*_{type}_{id}.out (DP/TP/EP tags)
      │
      └─ node_config: NodeConfig
         ├─ launch_command: NodeLaunchCommand
         │  └─ Source: 📁 logs/*_{type}_{id}.out (command line)
         ├─ environment: dict[str, str]
         │  └─ Source: 📁 logs/config.yaml
         └─ gpu_info: dict (optional)
            └─ Source: 📁 logs/*_config.json
```

---

## 8. Parser Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Parser Registry System                            │
│                                                                          │
│  Decorators:                                                            │
│  • @register_benchmark_parser("sa-bench")                               │
│  • @register_benchmark_parser("mooncake-router")                        │
│  • @register_node_parser("sglang")                                      │
│  • @register_node_parser("trtllm")                                      │
│                                                                          │
│  Lookup Functions:                                                      │
│  • get_benchmark_parser(type) → BenchmarkParser                         │
│  • get_node_parser(type) → NodeParser                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴────────────────┐
                    │                                │
                    ▼                                ▼
        ┌──────────────────────┐       ┌──────────────────────┐
        │  BenchmarkParsers    │       │    NodeParsers       │
        └──────────────────────┘       └──────────────────────┘
                    │                                │
        ┌───────────┴───────────┐        ┌──────────┴──────────┐
        ▼                       ▼        ▼                     ▼
┌──────────────┐    ┌──────────────┐  ┌─────────┐   ┌──────────────┐
│  SABench     │    │  Mooncake    │  │ SGLang  │   │   TRT-LLM    │
│   Parser     │    │   Parser     │  │ Parser  │   │   Parser     │
└──────────────┘    └──────────────┘  └─────────┘   └──────────────┘

Each parser implements:
  Benchmark:
    • find_result_directory() - locate result files
    • parse_result_directory() - parse all results
    • parse_result_json() - parse single result file
    • parse_launch_command() - extract command

  Node:
    • parse_logs() - parse directory of logs
    • parse_single_log() - parse one worker log
    • parse_launch_command() - extract command
```

---

## 9. Parsing Strategy: JSON-First Approach

### Design Principle: JSON as Source of Truth ✨

The parser infrastructure follows a **JSON-first** approach for benchmark results:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Benchmark Result Parsing Priority                     │
│                                                                          │
│  1️⃣ PRIMARY: JSON Result Files (Source of Truth)                        │
│     📁 result_*.json (SA-Bench)                                         │
│     📁 profile_export_aiperf.json (Mooncake-Router)                     │
│     - Complete, structured data                                         │
│     - Machine-readable, validated format                                │
│     - Contains all metrics with precision                               │
│                                                                          │
│  2️⃣ FALLBACK: benchmark.out Parsing                                     │
│     📁 logs/benchmark.out                                               │
│     - Used ONLY when JSON files are unavailable                         │
│     - Regex-based extraction from human-readable logs                   │
│     - May be incomplete or imprecise                                    │
│     - Logged as fallback in parser output                               │
└─────────────────────────────────────────────────────────────────────────┘
```

### Implementation

All benchmark parsers implement this strategy in `parse_result_directory()`:

```python
def parse_result_directory(self, result_dir: Path) -> list[dict[str, Any]]:
    results = []
    
    # 1️⃣ PRIMARY: Try JSON files first
    for json_file in result_dir.glob("*.json"):  # or rglob() for nested
        result = self.parse_result_json(json_file)
        if result.get("output_tps"):
            results.append(result)
            logger.info(f"Loaded from JSON: {json_file}")
    
    # 2️⃣ FALLBACK: If no JSON found, try benchmark.out
    if not results:
        benchmark_out = result_dir / "benchmark.out"
        if benchmark_out.exists():
            logger.info("No JSON results found, falling back to .out parsing")
            fallback_result = self.parse(benchmark_out)
            if fallback_result.get("output_tps"):
                results.append(fallback_result)
        else:
            logger.warning(f"No results found in {result_dir}")
    
    return results
```

### Rationale

1. **Accuracy**: JSON files contain exact, validated data
2. **Completeness**: JSON includes all metrics, not just what's in logs
3. **Reliability**: Structured format vs regex parsing
4. **Performance**: JSON parsing is faster than regex on large logs
5. **Maintainability**: Less brittle than log format changes

### When Fallback is Used

The fallback to `.out` file parsing occurs when:
- JSON result files are missing (incomplete benchmark run)
- Results directory doesn't contain expected JSON files
- Legacy runs from before JSON export was implemented

---

## 10. Caching Layer

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CacheManager                                   │
│                                                                          │
│  Caches to 📁 {run_path}/cached_assets/                                 │
│                                                                          │
│  Cached Data:                                                           │
│  • benchmark_results.parquet - ProfilerResults                          │
│  • node_metrics.parquet - NodeMetrics (all workers)                     │
│  • cache_metadata.json - timestamps, source patterns                    │
│                                                                          │
│  Cache Validation:                                                      │
│  • Checks if source files have changed (mtime)                          │
│  • Invalidates cache if patterns don't match                            │
│  • Automatically rebuilds if invalid                                    │
└─────────────────────────────────────────────────────────────────────────┘

Flow with cache:
  1. RunLoader checks cache validity
  2. If valid → deserialize from .parquet
  3. If invalid → parse from source files → cache results
  4. Populate NodeConfig from files (not cached)
```

---

## 11. File Structure Summary

```
{run_directory}/
├── metadata.json              → RunMetadata
├── config.yaml                → ProfilerMetadata.isl/osl
├── logs/
│   ├── benchmark.out          → BenchmarkLaunchCommand, ProfilerMetadata, (fallback metrics)
│   ├── config.yaml            → NodeConfig.environment
│   ├── {node}_{type}_{id}.out → NodeMetrics, NodeLaunchCommand
│   ├── {node}_config.json     → NodeConfig.gpu_info
│   └── sa-bench_isl_*/
│       └── result_*.json      → ProfilerResults (PRIMARY ✨)
│   └── artifacts/
│       └── */
│           └── profile_export_aiperf.json → ProfilerResults (PRIMARY ✨)
└── cached_assets/
    ├── benchmark_results.parquet
    ├── node_metrics.parquet
    └── cache_metadata.json
```

**Note**: JSON files are the primary source of truth for benchmark results.
The `.out` files serve as fallback for legacy/incomplete runs.

---

## 12. Key Design Principles

1. **Parser Autonomy**: Each parser knows how to find and parse its own files
   - `find_result_directory()` encapsulates file discovery logic
   - RunLoader doesn't need benchmark-specific knowledge

2. **JSON-First Parsing** ✨: JSON files are the primary source of truth
   - `parse_result_json()` for structured, accurate data
   - `parse()` method is fallback for when JSON is unavailable
   - Logged clearly when fallback is used

3. **Separation of Concerns**:
   - **Metrics** (NodeMetrics): Performance data from log parsing
   - **Configuration** (NodeConfig): Launch commands, environment, GPU info
   - **Metadata** (NodeMetadata): Worker identification

4. **Caching Strategy**:
   - Cache expensive parsing operations (batch/memory metrics)
   - Don't cache configuration (files are small, may change)
   - Validate cache against source file timestamps

5. **Extensibility**:
   - New benchmark types: Implement BenchmarkParserProtocol
   - New node backends: Implement NodeParserProtocol
   - Register with decorator → automatically available

6. **Data Flow Direction**:
   ```
   JSON Files (Primary) ──┐
                          ├──► Parsers ──► Data Models ──► Cache ──► Application
   .out Files (Fallback) ─┘
   ```

---

## 12. Usage Example

```python
from pathlib import Path
from analysis.srtlog.run_loader import RunLoader

# Load a run
loader = RunLoader("/path/to/runs")
run = loader.load_single("553")

# Access metadata (from metadata.json)
print(f"Job: {run.metadata.job_id}")
print(f"Model: {run.metadata.model.path}")

# Access profiler results (from result_*.json or profile_export_aiperf.json)
print(f"Output TPS: {run.profiler.output_tps}")
print(f"Mean TTFT: {run.profiler.mean_ttft_ms}")

# Access benchmark launch command (from logs/benchmark.out)
print(f"Benchmark: {run.benchmark_launch_command.benchmark_type}")
print(f"Arguments: {run.benchmark_launch_command.extra_args}")

# Load node metrics (from logs/*_{type}_{id}.out)
nodes = loader.load_node_metrics_for_run(run)
for node in nodes:
    # Metrics from log file parsing
    print(f"Node: {node.node_name} ({node.worker_type})")
    print(f"  Batches: {len(node.batches)}")
    print(f"  Memory snapshots: {len(node.memory_snapshots)}")
    
    # Config from config files
    print(f"  Backend: {node.launch_command.backend_type}")
    print(f"  Environment vars: {len(node.environment)}")
```


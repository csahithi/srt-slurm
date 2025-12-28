# Exemplar Add Worklog

## Goal
Implement the qwen3-32b exemplar from dynamo as a new benchmark called "mooncake-router" in infbench.

## Progress

### Phase 1: Research & Understanding
- [x] Explore dynamo recipes folder structure
- [x] Read the README to understand the benchmark
- [x] Examine k8s yamls for vLLM recipes
- [x] Understand sglang integration requirements

### Phase 2: Implementation
- [x] Create YAML configs for qwen32b with sglang
- [x] Implement mooncake-router benchmark runner
- [x] Create benchmark scripts (using aiperf like dynamo)
- [x] Add BenchmarkType enum entry
- [x] Register in __init__.py
- [x] Lint passes

---

## Summary of Changes

### New Files Created

1. **`recipies/qwen3-32b/disagg-kv-router.yaml`**
   - Disaggregated mode: 6 prefill + 2 decode workers with TP2
   - Uses cache-aware routing policy
   - 16x H200 GPUs across 2 nodes

2. **`recipies/qwen3-32b/agg-round-robin.yaml`**
   - Aggregated mode: 8x TP2 workers
   - Uses round-robin routing policy (baseline)
   - 16x H200 GPUs across 2 nodes

3. **`src/srtctl/benchmarks/mooncake_router.py`**
   - Benchmark runner class `MooncakeRouterRunner`
   - Registered as "mooncake-router" benchmark type
   - Validates workload type and threshold parameters

4. **`src/srtctl/benchmarks/scripts/mooncake-router/bench.sh`**
   - Uses `aiperf profile` exactly like dynamo does
   - Downloads Mooncake trace files (conversation, synthetic, toolagent, mooncake)
   - Runs with `--custom-dataset-type mooncake_trace --fixed-schedule`
   - Supports goodput thresholds for TTFT and ITL

### Modified Files

1. **`src/srtctl/core/schema.py`**
   - Added `MOONCAKE_ROUTER = "mooncake-router"` to BenchmarkType enum
   - Added benchmark fields: `mooncake_workload`, `ttft_threshold_ms`, `itl_threshold_ms`

2. **`src/srtctl/benchmarks/__init__.py`**
   - Added import for `mooncake_router`
   - Added to `__all__` list

---

## Log

### Entry 1 - Initial Exploration (Research Complete)

**Dynamo Exemplar Overview:**
- Location: `/Users/idhanani/Desktop/dyn/dynamo/recipes/qwen3-32b/`
- Two deployment modes:
  1. **Aggregated (round-robin)**: 8x TP2 workers on 16 GPUs
  2. **Disaggregated (KV-aware)**: 6 prefill + 2 decode (TP2) with KV-aware routing

**Benchmark Dataset:**
- Uses Mooncake conversation trace from FAST25 paper
- 12,031 requests over ~59 minutes (3.4 req/s)
- Avg input: 12,035 tokens, Avg output: 343 tokens
- 36.64% cache efficiency potential

**Key vLLM Config (from deploy.yaml):**
- Model: Qwen/Qwen3-32B
- TP size: 2
- GPU memory util: 0.90
- Block size: 64
- Max model len: 131072
- Rope scaling with yarn

**Benchmark Tool (from perf.yaml):**
- Uses `aiperf profile` command
- Key arguments:
  - `-m ${MODEL_NAME}`
  - `--input-file ${INPUT_FILE}` (mooncake trace jsonl)
  - `--custom-dataset-type mooncake_trace`
  - `--fixed-schedule` (replays at original timestamps)
  - `--streaming`
  - `--goodput "time_to_first_token:2000 inter_token_latency:25"`

### Entry 2 - Implementation Complete

Created all files matching the dynamo exemplar pattern but adapted for sglang backend instead of vLLM. The benchmark script uses aiperf exactly as dynamo does, ensuring compatibility with the Mooncake trace format and fixed-schedule replay.

**Usage:**
```bash
# Run disaggregated KV-aware routing benchmark
srtctl submit -f recipies/qwen3-32b/disagg-kv-router.yaml

# Run aggregated round-robin baseline
srtctl submit -f recipies/qwen3-32b/agg-round-robin.yaml
```

The benchmark will:
1. Download the Mooncake conversation trace
2. Wait for model to be ready
3. Run aiperf with fixed-schedule replay
4. Report goodput metrics (TTFT, ITL thresholds)

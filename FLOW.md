# Complete Job Submission Flow

This document traces the entire flow from `uv run srtctl config.yaml` to running SGLang workers.

## Example Command

```bash
uv run srtctl configs/gb200_fp8_1p_4d.yaml
```

---

## Phase 1: CLI Entry & Config Loading

### Files: `src/srtctl/cli/submit.py`, `src/srtctl/core/config.py`, `src/srtctl/core/schema.py`

1. **CLI Entry**
   - User runs: `uv run srtctl configs/gb200_fp8_1p_4d.yaml`
   - Entry point: `srtctl.cli.submit:main()`
   - Parses args → calls `submit_single(config_path)`

2. **Load & Validate Config**
   ```python
   config = load_config(Path("configs/gb200_fp8_1p_4d.yaml"))
   ```
   - Reads YAML file
   - Loads cluster defaults from `srtslurm.yaml` (if exists)
     - Applies default account, partition, time_limit
     - Resolves model path/container aliases
   - Validates with pydantic `JobConfig` schema
     - Checks required fields
     - Validates enum values (gpu_type, precision)
     - Auto-populates `backend.gpu_type = "gb200-fp8"` from `resources.gpu_type` + `model.precision`
   - Returns validated dict

---

## Phase 2: Backend Setup & File Generation

### File: `src/srtctl/backends/sglang.py`

3. **Create Backend Instance**
   ```python
   backend = SGLangBackend(config)
   ```

4. **Generate SGLang Config YAML**
   ```python
   sglang_config_path = backend.generate_config_file()
   ```

   - Extracts `sglang_config.prefill` and `sglang_config.decode` from config
   - Converts underscore keys to dashes:
     - `kv_cache_dtype` → `kv-cache-dtype`
     - `mem_fraction_static` → `mem-fraction-static`
   - Creates temp file: `/tmp/sglang_config_xyz.yaml`:
     ```yaml
     prefill:
       kv-cache-dtype: fp8_e4m3
       mem-fraction-static: 0.95
       disaggregation-mode: prefill
       # ... all other flags
     decode:
       kv-cache-dtype: fp8_e4m3
       mem-fraction-static: 0.95
       disaggregation-mode: decode
       # ... all other flags
     ```
   - Returns path to temp file

5. **Generate SLURM Job Script**
   ```python
   script_path, rendered_script = backend.generate_slurm_script(
       config_path=sglang_config_path,
       timestamp="20251118_233054"
   )
   ```

   - Determines mode: disaggregated (1P+4D) or aggregated
   - Selects template:
     - Disaggregated: `scripts/templates/job_script_template_disagg.j2`
     - Aggregated: `scripts/templates/job_script_template_agg.j2`
   - Builds template variables:
     ```python
     template_vars = {
         "job_name": "gb200-fp8-1p-4d",
         "total_nodes": 5,  # 1 prefill + 4 decode
         "prefill_nodes": 1,
         "decode_nodes": 4,
         "prefill_workers": 1,
         "decode_workers": 1,
         "gpus_per_node": 4,
         "gpu_type": "gb200-fp8",
         "script_variant": "max-tpt",  # Legacy script to use
         "sglang_torch_profiler": False,  # Or True if enable_profiling
         # ... many more
     }
     ```
   - Renders Jinja2 template → bash script
   - Writes to temp file: `/tmp/slurm_job_xyz.sh`
   - Returns `(script_path, rendered_script_content)`

---

## Phase 3: SLURM Submission

### File: `src/srtctl/cli/submit.py`

6. **Submit to SLURM**
   ```python
   result = subprocess.run(["sbatch", str(script_path)], ...)
   ```

   - SLURM receives script
   - Allocates 5 nodes (1 prefill + 4 decode)
   - Returns: `"Submitted batch job 3667"`
   - Parse job ID: `3667`

7. **Create Log Directory**
   ```
   ../srtctl/logs/3667_1P_4D_20251118_233054/
   ```

   Files saved:
   - `sbatch_script.sh` - The generated SLURM script
   - `config.yaml` - Resolved config with all defaults
   - `sglang_config.yaml` - SGLang flags (prefill/decode)
   - `jobid.json` - Metadata (job ID, resources, benchmark config)

---

## Phase 4: SLURM Job Execution

### File: Generated script from `scripts/templates/job_script_template_disagg.j2`

8. **Job Starts on Allocated Nodes**

   SLURM allocates:
   - Node 0: Nginx (load balancer)
   - Node 1: Master (ETCD, NATS, Frontend 0)
   - Node 2-4: Additional frontends (if `enable_multiple_frontends=true`)
   - Remaining nodes: Prefill/decode workers

9. **Infrastructure Setup (Master Node Only)**

   The master node starts:
   - **ETCD** (port 2379) - Distributed key-value store for coordination
   - **NATS** (port 4222) - Message broker for worker communication

   Command:
   ```bash
   /configs/nats-server -js &  # Or nats-server if not using dynamo wheels
   /configs/etcd --listen-client-urls http://0.0.0.0:2379 ... &
   ```

10. **Wait for Infrastructure**

    All nodes wait for ETCD to be healthy:
    ```bash
    while ! curl -s http://master_ip:2379/health; do sleep 2; done
    ```

---

## Phase 5: Worker Launch

### File: `scripts/worker_setup.py`

For EACH worker node, the sbatch script runs:

```bash
srun ... python /scripts/worker_setup.py \
    --worker-type prefill \  # or "decode"
    --worker-idx 0 \
    --local-rank 0 \
    --leader-ip 10.0.0.1 \
    --nodes-per-worker 1 \
    --gpus-per-node 4 \
    --gpu-type gb200-fp8 \
    --script-variant max-tpt \
    --sglang-torch-profiler \  # Only if enable_profiling=true
    --dump-config-path /logs/node_config.json \
    --use-dynamo-whls \
    --use-init-locations \
    ...
```

11. **Worker Setup (`worker_setup.py`)**

    a. **Install Dependencies** (if `--use-dynamo-whls`):
       ```bash
       pip install /configs/ai_dynamo_runtime-*.whl
       pip install /configs/ai_dynamo-*.whl
       ```

    b. **Set Environment Variables**:
       ```python
       os.environ["HOST_IP_MACHINE"] = "10.0.0.1"
       os.environ["PORT"] = "29500"
       os.environ["TOTAL_GPUS"] = "4"  # nodes_per_worker * gpus_per_node
       os.environ["RANK"] = "0"  # local_rank
       os.environ["TOTAL_NODES"] = "1"
       os.environ["USE_INIT_LOCATIONS"] = "True"
       os.environ["USE_DYNAMO_WHLS"] = "True"
       os.environ["USE_SGLANG_LAUNCH_SERVER"] = "False"  # True if profiling
       os.environ["SGLANG_TORCH_PROFILER_DIR"] = "/logs/profiles/prefill"  # If profiling
       ```

    c. **Start GPU Utilization Monitoring**:
       ```bash
       bash /scripts/monitor_gpu_utilization.sh > /logs/gpu_util.log &
       ```

    d. **Get GPU Command**:
       ```python
       cmd = get_gpu_command("prefill", "gb200-fp8", "max-tpt")
       # Returns: "bash /scripts/legacy/gb200-fp8/disagg/max-tpt.sh prefill"
       ```

    e. **Execute Command**:
       ```python
       return run_command(cmd)  # Blocks until worker exits
       ```

---

## Phase 6: Legacy Script Execution

### File: `scripts/legacy/gb200-fp8/disagg/max-tpt.sh`

12. **Legacy Script Runs**

    The script receives mode as arg: `bash max-tpt.sh prefill`

    a. **Determine Python Module**:
       ```bash
       if [[ "${USE_SGLANG_LAUNCH_SERVER,,}" == "true" ]]; then
           PYTHON_MODULE="sglang.launch_server"  # Profiling mode
       else
           PYTHON_MODULE="dynamo.sglang"  # Normal mode
       fi
       ```

    b. **Check Environment Variables**:
       ```bash
       # Validates: HOST_IP_MACHINE, PORT, TOTAL_GPUS, RANK, etc.
       ```

    c. **Install Wheels** (if needed):
       ```bash
       if [[ "${USE_DYNAMO_WHLS,,}" == "true" ]]; then
           python3 -m pip install /configs/ai_dynamo_*.whl
       fi
       ```

    d. **Build SGLang Command**:
       ```bash
       # For prefill mode:
       command_suffix=""
       if [[ "${USE_INIT_LOCATIONS,,}" == "true" ]]; then
           command_suffix="--init-expert-location /configs/prefill_dsr1.json"
       fi
       if [[ -n "${DUMP_CONFIG_PATH}" ]]; then
           command_suffix="${command_suffix} --dump-config-to ${DUMP_CONFIG_PATH}"
       fi

       # Skip --disaggregation-mode in profiling
       if [[ "${USE_SGLANG_LAUNCH_SERVER,,}" != "true" ]]; then
           DISAGG_MODE_FLAG="--disaggregation-mode prefill"
       else
           DISAGG_MODE_FLAG=""
       fi
       ```

    e. **Execute SGLang**:
       ```bash
       SGLANG_ENABLE_JIT_DEEPGEMM=false \
       SGLANG_ENABLE_FLASHINFER_GEMM=1 \
       ... (many more env vars) ...
       python3 -m dynamo.sglang \
           ${DISAGG_MODE_FLAG} \
           --served-model-name deepseek-ai/DeepSeek-R1 \
           --model-path /model/ \
           --kv-cache-dtype fp8_e4m3 \
           --mem-fraction-static 0.95 \
           --quantization fp8 \
           --moe-runner-backend flashinfer_trtllm \
           --attention-backend trtllm_mla \
           --disable-radix-cache \
           ... (50+ more flags from script) ...
           --dist-init-addr ${HOST_IP_MACHINE}:${PORT} \
           --nnodes ${TOTAL_NODES} \
           --node-rank ${RANK} \
           --ep-size 4 \
           --tp-size 4 \
           --dp-size 4 \
           ${command_suffix}
       ```

**Note**: The SGLang flags in the legacy script are mostly ignored! The actual flags come from the SGLang config YAML we generated. But wait... how?

---

## The Missing Piece: How Does YAML Config Get Used?

**This is the current gap in the implementation!**

The legacy scripts have hardcoded SGLang flags, but YAML configs want to use the generated `sglang_config.yaml`.

### Current Reality:
- YAML configs generate `sglang_config.yaml` with user's flags
- But legacy scripts run with their own hardcoded flags
- The YAML config is ignored!

### Needed Solution:
Either:
1. Create new `yaml-config.sh` scripts that read from `sglang_config.yaml`, OR
2. Modify worker_setup.py to execute commands directly for YAML configs, OR
3. Accept that YAML configs still use legacy scripts (user provides `script_variant`)

For now, **option 3** is what happens - users must specify a `script_variant` that matches their desired configuration, or we default to `max-tpt`.

---

## Phase 7: SGLang Workers Running

13. **Prefill Worker Starts**
    - Connects to ETCD at `master_ip:2379`
    - Loads model from `/model/` with FP8 quantization
    - Initializes KV cache (FP8_E4M3)
    - Listens on port 30000 for requests
    - Registers with ETCD

14. **Decode Workers Start** (4 nodes)
    - Each connects to ETCD
    - Each connects to prefill worker via ETCD discovery
    - Load model (FP8 quantization)
    - Initialize KV cache
    - Wait for prefill to send KV data

15. **Nginx Load Balancer** (if `enable_multiple_frontends=true`)
    - Starts on node 0
    - Configuration:
      ```nginx
      upstream sglang_backends {
          server node1:30000;
          server node2:30000;
          server node3:30000;
          # ... all frontend nodes
      }
      ```
    - Accepts requests on port 30000
    - Load balances across all frontends

---

## Phase 8: Benchmarking

### File: `scripts/benchmarks/sa-bench.sh`

16. **If `benchmark.type = "sa-bench"`**:

    The sbatch script runs:
    ```bash
    bash /scripts/benchmarks/sa-bench.sh \
        1024 \  # isl
        1024 \  # osl
        256x512  # concurrencies
    ```

    Steps:
    - Wait for server health: `curl http://master:30000/health`
    - For each concurrency level (256, 512):
      ```bash
      python3 -m sglang.bench_serving \
          --backend sglang \
          --base-url http://master:30000 \
          --tokenizer /model/ \
          --dataset-name random \
          --num-prompts 1000 \
          --random-input-len 1024 \
          --random-output-len 1024 \
          --request-rate inf \
          --multi 256 \  # concurrency
          --output-file /logs/sa_bench_256.json
      ```
    - Results saved to `/logs/sa_bench_results.json`

17. **If `benchmark.type = "manual"`**:
    - Workers just run indefinitely
    - User sends requests manually
    - Good for profiling or interactive testing

---

## Key Insights

### Environment Variable Flow

```
YAML Config
    ↓
Backend (generate_slurm_script)
    ↓
Template Variables (sglang_torch_profiler, etc.)
    ↓
SLURM Script
    ↓
worker_setup.py args (--sglang-torch-profiler)
    ↓
Environment Variables (USE_SGLANG_LAUNCH_SERVER=True)
    ↓
Legacy Script (max-tpt.sh)
    ↓
Conditional: python3 -m sglang.launch_server vs dynamo.sglang
```

### Config Precedence

1. **User YAML** - Explicit user config
2. **Cluster Defaults** (srtslurm.yaml) - Account, partition, defaults
3. **Pydantic Defaults** - Schema defaults (e.g., `enable_profiling=False`)
4. **Auto-populated** - `backend.gpu_type` from `resources + model`
5. **Template Defaults** - Hardcoded in Jinja templates
6. **Legacy Script** - Hardcoded in bash scripts (currently used)

### The YAML Config Solution

**Solution**: Use dynamo.sglang's native `--config` flag support!

Instead of parsing YAML and building 50+ CLI flags, we simply:
```bash
python3 -m dynamo.sglang \
    --config /path/to/sglang_config.yaml \
    --config-key prefill \
    --dist-init-addr $HOST_IP_MACHINE:$PORT \
    --nnodes 1 \
    --node-rank $RANK
```

**How It Works**:
1. `worker_setup.py` receives `--sglang-config-path` argument
2. When sglang_config_path is provided, `get_gpu_command()` calls `build_sglang_command_from_yaml()`
3. This builds a simple command using `--config` and `--config-key` flags
4. Only coordination flags (dist-init-addr, nnodes, node-rank) are added dynamically
5. All SGLang-specific flags (ep-size, tp-size, dp-size, kv-cache-dtype, etc.) come from YAML
6. Legacy scripts are still available as fallback when sglang_config_path is not provided

---

## File Flow Diagram

```
User YAML Config (configs/gb200_fp8_1p_4d.yaml)
    ↓
load_config() → Validates with pydantic
    ↓
SGLangBackend.generate_config_file()
    ↓
Temp SGLang YAML (/tmp/sglang_config_xyz.yaml)
    |
    ├── prefill: {kv-cache-dtype: fp8_e4m3, ...}
    └── decode: {kv-cache-dtype: fp8_e4m3, ...}
    ↓
SGLangBackend.generate_slurm_script()
    ↓
Jinja2 Template (job_script_template_disagg.j2)
    ↓
Rendered SLURM Script (/tmp/slurm_job_xyz.sh)
    ↓
sbatch submission
    ↓
SLURM Execution on Cluster
    ↓
Per-Node: worker_setup.py
    ↓
Legacy Script: scripts/legacy/gb200-fp8/disagg/max-tpt.sh
    ↓
SGLang Process: python3 -m dynamo.sglang ...
```

---

## Completed YAML Support Implementation

✅ **worker_setup.py updates**:
   - Added `--sglang-config-path` CLI argument
   - Created `build_sglang_command_from_yaml()` function
   - Updated `get_gpu_command()` to support YAML configs
   - Updated all worker setup functions (prefill, decode, aggregated) to accept and use sglang_config_path
   - Maintains backward compatibility with legacy bash scripts

## Remaining Steps

1. **Update SLURM templates**:
   - Modify `scripts/templates/job_script_template_disagg.j2` to pass `--sglang-config-path` to worker_setup.py
   - Modify `scripts/templates/job_script_template_agg.j2` similarly
   - The SGLang config file needs to be copied to a location accessible from worker nodes (e.g., /logs/ directory)

2. **Update config file handling in backend**:
   - Ensure the generated sglang_config.yaml is saved to the log directory
   - Pass the correct path to the SLURM template

3. **Test end-to-end**:
   - Submit test job with YAML config
   - Verify command uses `--config` flag
   - Test both normal and profiling modes
   - Verify all flags from YAML are applied correctly

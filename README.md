# SRT Slurm

Benchmarking toolkit for Dynamo and SGLang on SLURM.

## Run a benchmark

1. Run `make setup` to download the neccesary dynamo dependencies. This pulls in `nats` and `etcd` and the dynamo pip wheels. This allows you to use any container found on the [lmsys dockerhub](https://hub.docker.com/r/lmsysorg/sglang/tags) and use dynamo orchestration.

2. Run your very first benchmark using the following python command. WIP to make this command much much shorter and possibly hold some of the common pieces in a configuration file.

```bash
python3 submit_job_script.py \
  --model-dir /mnt/lustre01/models/deepseek-r1-0528-fp4-v2 \
  --container-image /mnt/lustre01/users/slurm-shared/ishan/lmsysorg+sglang+v0.5.5.post2.sqsh \
  --gpus-per-node 4 \
  --config-dir /mnt/lustre01/users/slurm-shared/ishan/srt-slurm/configs \
  --gpu-type gb200-fp4 \
  --network-interface enP6p9s0np0 \
  --prefill-nodes 1 \
  --decode-nodes 12 \
  --prefill-workers 1 \
  --decode-workers 1 \
  --account nvidia \
  --partition batch \
  --time-limit 4:00:00 \
  --enable-multiple-frontends \
  --num-additional-frontends 9 \
  --benchmark "type=sa-bench; isl=1024; osl=1024; concurrencies=1x8x32x128x512x1024x2048x4096x8192; req-rate=inf" \
  --script-variant max-tpt \
  --use-dynamo-whls \
  --log-dir /mnt/lustre01/users-public/slurm-shared/joblogs
```

For more info on the submission script see [slurm_jobs/README.md](slurm_jobs/README.md)

## Run the UI

```bash
./run_dashboard.sh
```

The dashboard will open at http://localhost:8501 and scan the current directory for benchmark runs. You can specify your own log directory in the UI itself.

## What It Does

**Pareto Analysis** - Compare throughput efficiency (TPS/GPU) vs per-user throughput (TPS/User) across configurations

**Latency Breakdown** - Visualize TTFT, TPOT, and ITL metrics as concurrency increases

**Config Comparison** - View deployment settings (TP/DP) and hardware specs side-by-side

**Data Export** - Sort, filter, and export metrics to CSV

## Key Metrics

- **Output TPS/GPU** - Throughput per GPU (higher = more efficient)
- **Output TPS/User** - Throughput per concurrent user (higher = better responsiveness)
- **TTFT** - Time to first token (lower = faster start)
- **TPOT** - Time per output token (lower = faster generation)
- **ITL** - Inter-token latency (lower = smoother streaming)

## Directory Structure

This structure comes built into the scripts. WIP to handle other directory structures.

The app expects benchmark runs in subdirectories with:

- `{jobid}.json` - Metadata file with run configuration (required)
- `vllm_isl_*_osl_*/` containing `*.json` result files
- `*_config.json` files for node configurations
- `*_prefill_*.err` and `*_decode_*.err` files for node metrics

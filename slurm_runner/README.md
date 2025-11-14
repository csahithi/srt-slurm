# SLURM Job Submission

Scripts for submitting benchmark jobs to SLURM clusters.

## Prerequisites

1. **SLURM cluster** with GPU nodes
2. **Pyxis plugin** for container support (`--container-image`, `--container-mounts`, etc.)
3. **Container image** with Dynamo+SGLang ([build instructions](https://hub.docker.com/r/lmsysorg/sglang/tags))

## Quick Start

1. **Run setup** (from repo root):

   ```bash
   make setup
   ```

   This creates `srtslurm.yaml` with your cluster defaults.

2. **Submit a job**:

   ```bash
   cd slurm_runner
   python3 submit_job_script.py \
     --model-dir /path/to/model \
     --gpu-type gb200-fp4 \
     --gpus-per-node 4 \
     --prefill-nodes 1 \
     --decode-nodes 12 \
     --prefill-workers 1 \
     --decode-workers 1 \
     --script-variant max-tpt \
     --benchmark "type=sa-bench; isl=1024; osl=1024; concurrencies=1024x2048; req-rate=inf"
   ```

   All cluster settings (account, partition, network interface, container image) are read from `../srtslurm.yaml`.

3. **Monitor logs**:
   ```bash
   cd ../logs/{JOB_ID}_*
   tail -f *_prefill_*.err *_decode_*.err
   ```

## Configuration

### Cluster Settings (`srtslurm.yaml`)

Created automatically by `make setup` in repo root:

```yaml
cluster:
  account: "restricted"
  partition: "batch"
  network_interface: "enP6p9s0np0"
  time_limit: "4:00:00"
  container_image: "/path/to/container.sqsh"
```

All settings can be overridden via CLI flags.

### Config Directory

The `--config-dir` (defaults to `../configs`) contains:

- **Dynamo wheels**: `ai_dynamo*.whl` (downloaded by `make setup`)
- **Binaries**: `nats-server`, `etcd`, `etcdctl` (downloaded by `make setup`)
- **DeepEP config**: `deepep_config.json` (optional, for GB200)
- **Expert locations**: `*_init-expert-location.json` (optional, use `--use-init-location`)
- **DeepGEMM cache**: `dgcache/` directory (optional)

### Script Variants

Script variants are shell scripts in `scripts/<gpu-type>/<mode>/`:

- `max-tpt.sh` - Maximum throughput configuration
- `1p_4d.sh` - 1 prefill, 4 decode workers
- Custom variants - Add your own `.sh` files

Available GPU types: `gb200-fp4`, `gb200-fp8`

## Optional Arguments

These have sensible defaults or read from `srtslurm.yaml`:

```bash
--account <account>              # From srtslurm.yaml
--partition <partition>          # From srtslurm.yaml
--network-interface <interface>  # From srtslurm.yaml
--time-limit <HH:MM:SS>         # From srtslurm.yaml or 04:00:00
--container-image <path>         # From srtslurm.yaml
--config-dir <path>              # Defaults to ../configs
--log-dir <path>                 # Defaults to ../logs
```

## Logs

Jobs create directories named: `{SLURM_JOB_ID}_{P}P_{D}D_{TIMESTAMP}/`

Example: `3667_1P_12D_20251113_214831/`

Each contains:

- `{JOB_ID}.json` - Run metadata
- `sa-bench_isl_*/` - Benchmark results (JSON)
- `*_config.json` - Node configurations
- `*.err`, `*.out` - SLURM logs

## Benchmarking

Specify benchmark config with `--benchmark`:

```bash
--benchmark "type=sa-bench; isl=1024; osl=1024; concurrencies=128x512x1024; req-rate=inf"
```

- `type`: Benchmark type (`sa-bench`, `sglang`, `manual`)
- `isl`: Input sequence length
- `osl`: Output sequence length
- `concurrencies`: Concurrency levels (x-separated)
- `req-rate`: Request rate (`inf` for max throughput)

Set `type=manual` to skip automated benchmarking.

## Advanced

### Multiple Frontends

Multiple frontends with nginx load balancing are **enabled by default** (9 additional frontends).

To disable:

```bash
--disable-multiple-frontends
```

To change the count:

```bash
--num-additional-frontends 5
```

### Aggregated Mode

For non-disaggregated deployments:

```bash
--agg-nodes 4 \
--agg-workers 1
```

(Use `--agg-*` instead of `--prefill-*` and `--decode-*`)

## Troubleshooting

**Network interface issues**: The templates use `ip addr show $NETWORK_INTERFACE` to find node IPs. If this doesn't work on your cluster, modify the templates or set your interface correctly.

**Missing configs**: Ensure you ran `make setup` from repo root to download required binaries.

For full documentation, see [main README](../README.md).

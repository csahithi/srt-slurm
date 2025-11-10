# SRT Slurm Benchmark Dashboard

Interactive Streamlit dashboard for visualizing and analyzing end to end sglang benchmarks run on SLURM clusters.

> [!NOTE]
> You must use the slurm jobs folder in the dynamo repository to run the job so that this benchmarking tools can analyze it

## Quick Start

```bash
./run_dashboard.sh
```

The dashboard will open at http://localhost:8501 and scan the current directory for benchmark runs.

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

## Installation

**With uv (recommended):**

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run the dashboard (uv handles dependencies automatically)
./run_dashboard.sh
```

## Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Setup pre-commit hooks
pre-commit install

# Run ruff manually
pre-commit run --all-files

# Or run ruff directly
uv run ruff check .
uv run ruff format .
```

## Directory Structure

The app expects benchmark runs in subdirectories with:

- `vllm_isl_*_osl_*/` containing `*.json` result files
- `*_config.json` files for node configurations

See `LOG_STRUCTURE.md` for detailed format.

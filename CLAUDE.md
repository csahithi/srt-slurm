# CLAUDE.md

Development guide for working on this codebase.

## Quick Reference

```bash
# Run all tests
uv run pytest tests/ -v

# Run single test file
uv run pytest tests/test_e2e.py -v

# Run single test
uv run pytest tests/test_e2e.py::TestH100Cluster::test_endpoint_allocation -v

# Lint and format
uv run ruff check --fix src/
uv run ruff format src/

# Type check
uv run ty check src/srtctl/

# Pre-commit (runs ruff lint + format)
uvx pre-commit run --all-files
```

## Code Style

- **Python 3.10+** - use modern syntax (`|` unions, `match` statements)
- **Ruff** for linting and formatting (config in `pyproject.toml`)
- **Type hints** everywhere - use `ty` for type checking
- **Frozen dataclasses** for configs (`@dataclass(frozen=True)`)
- **Line length**: 120 characters

## Key Concepts

### RuntimeContext

Single source of truth for computed paths. Created once at job start:

```python
runtime = RuntimeContext.from_config(config, job_id)
runtime.log_dir          # /path/to/logs/12345_1P_4D_...
runtime.head_node_ip     # 10.0.0.1
runtime.container_mounts # List of mount strings
```

### Endpoint Allocation

Maps logical workers to physical nodes/GPUs:

```python
endpoints = allocate_endpoints(
    num_prefill=2, num_decode=4, num_agg=0,
    gpus_per_prefill=8, gpus_per_decode=4, gpus_per_agg=0,
    gpus_per_node=8,
    available_nodes=("node0", "node1", "node2"),
)
# Returns List[Endpoint] with node assignments and GPU indices
```

### Health Checks

Two patterns for checking worker readiness:

```python
# Dynamo backend
check_dynamo_health(response_json, expected_prefill=2, expected_decode=4)

# SGLang router
check_sglang_router_health(response_json, expected_prefill=2, expected_decode=4)
```

For aggregated mode, pass `expected_prefill=0, expected_decode=num_agg`.

## Testing

### Test Categories

- `test_e2e.py` - Cluster-style tests with mocked SLURM
- `test_endpoint_allocation.py` - GPU allocation logic
- `test_health.py` - Health check parsing
- `test_configs.py` - YAML config loading
- `test_command_generation.py` - SGLang command building

### Mocking SLURM

```python
class H100Rack:
    NUM_NODES = 13
    GPUS_PER_NODE = 8

    @classmethod
    def slurm_env(cls):
        return {
            "SLURM_JOB_ID": "12345",
            "SLURM_NODELIST": "h100-[01-13]",
            ...
        }

with patch.dict(os.environ, H100Rack.slurm_env()):
    with patch("subprocess.run", H100Rack.mock_scontrol()):
        # Test code here
```

## Common Tasks

### Adding a New Backend

1. Create `backends/mybackend.py` with a dataclass implementing `BackendProtocol`
2. Add `build_worker_command(process, runtime)` method
3. Export from `backends/__init__.py`

### Adding a New Benchmark

1. Create `benchmarks/mybench.py` inheriting from `BenchmarkRunner`
2. Implement `run(config, log_dir)` method
3. Add bash script to `benchmarks/scripts/mybench/bench.sh`
4. Register in benchmark type mapping

## Debugging

### Check Generated Commands

```bash
srtctl dry-run -f config.yaml
```

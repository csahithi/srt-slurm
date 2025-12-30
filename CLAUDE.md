# CLAUDE.md

Development guide for working on this codebase.

## Quick Reference

```bash
# Run lint + tests (recommended)
make check

# Just lint
make lint

# Just tests
make test

# Run single test file
uv run pytest tests/test_e2e.py -v

# Run single test
uv run pytest tests/test_e2e.py::TestH100Cluster::test_endpoint_allocation -v

# Auto-fix lint issues
uv run ruff check --fix src/srtctl/
uv run ruff format src/srtctl/
```

## Code Style

- **Python 3.10+** - use modern syntax (`|` unions, `match` statements)
- **Ruff** for linting and formatting (config in `pyproject.toml`)
- **Type hints** everywhere - use `ty` for type checking
- **Frozen dataclasses** for configs (`@dataclass(frozen=True)`)
- **Line length**: 120 characters

## Python Patterns

Follow these patterns when extending the codebase:

- **Frozen dataclasses for config** - Use `@dataclass(frozen=True)` for all configuration objects. Immutability prevents accidental mutation and makes code easier to reason about.
- **Protocol over ABC** - Prefer `typing.Protocol` for interface definitions (see `BackendProtocol`). Enables duck typing without inheritance coupling.
- **marshmallow_dataclass for validation** - Combine dataclasses with marshmallow schemas for type-safe config loading with validation. Custom fields (e.g., `BackendConfigField`) handle polymorphic deserialization.
- **Factory classmethods** - Use `@classmethod` named `from_*` for construction (e.g., `RuntimeContext.from_config()`, `RunMetadata.from_json()`). Keep `__init__` simple.
- **TYPE_CHECKING guard** - Import type-only dependencies under `if TYPE_CHECKING:` to avoid circular imports. Use string annotations for forward refs.
- **Computed properties** - Use `@property` for derived values instead of storing computed state. See `ResourceConfig.gpus_per_prefill`, `RunMetadata.topology_label`.
- **Registry pattern** - Use decorators for extensible registration (`@register_benchmark("sa-bench")`). New implementations just decorate and import.
- **TypedDict for external data** - Use `TypedDict` for typing dicts from JSON/external sources where you can't control the structure.
- **Single source of truth** - Create context objects (like `RuntimeContext`) that compute all derived paths/values once at startup rather than recomputing.
- **testing** - when we make a new significant feature change, we should always add a new test

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

Tests are located in `tests/`. Run `make check` to run lint + all tests.

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

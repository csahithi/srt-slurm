# srtctl

Command-line tool for distributed LLM inference benchmarks on SLURM clusters using SGLang. Replace complex shell scripts and 50+ CLI flags with declarative YAML configuration.

## Quick Start

```bash
# One-time setup
make setup ARCH=aarch64  # or ARCH=x86_64

# Submit a job
uv run srtctl apply -f examples/example.yaml
```

## Documentation

**Full documentation:** https://srtctl.gitbook.io/srtctl-docs/

# CLI Reference

`srtctl` is the main command-line interface for submitting benchmark jobs to SLURM.

## Table of Contents

- [Quick Start](#quick-start)
- [Interactive Mode](#interactive-mode)
  - [Recipe Browser](#recipe-browser)
  - [Configuration Summary](#configuration-summary)
  - [Interactive Actions Menu](#interactive-actions-menu)
  - [sbatch Preview](#sbatch-preview)
  - [Parameter Modification](#parameter-modification)
  - [Sweep Preview](#sweep-preview)
  - [Submission Confirmation](#submission-confirmation)
  - [Workflow Examples](#workflow-examples)
- [Commands](#commands)
  - [srtctl apply](#srtctl-apply)
  - [srtctl dry-run](#srtctl-dry-run)
- [Output](#output)
- [Sweep Support](#sweep-support)
- [Tips](#tips)

---

## Quick Start

```bash
# Interactive mode - browse recipes, preview, and submit
srtctl

# Submit a job directly
srtctl apply -f recipes/gb200-fp8/sglang-1p4d.yaml

# Preview without submitting
srtctl dry-run -f config.yaml
```

## Interactive Mode

Running `srtctl` with no arguments launches an interactive TUI (Text User Interface) powered by Rich and Questionary:

```bash
srtctl
# or explicitly:
srtctl -i
```

Interactive mode is ideal for:
- Exploring available recipes without memorizing paths
- Previewing and tweaking configurations before submission
- Understanding what a sweep will expand to
- Quick experimentation and validation

### Recipe Browser

On launch, interactive mode scans the `recipes/` directory and presents recipes organized by subdirectory:

```
? Select a recipe:
  ── gb200-fp8 ──
    sglang-1p4d.yaml
    sglang-2p8d.yaml
    dynamo-router.yaml
  ── h100-fp8 ──
    baseline.yaml
    high-throughput.yaml
  ──────────────
  📁 Browse for file...
```

**Features:**
- Recipes grouped by parent directory for easy navigation
- Arrow keys to navigate, Enter to select
- "Browse for file..." option for configs outside `recipes/`
- If no recipes found, prompts for manual path entry

### Configuration Summary

After selecting a recipe, you'll see a tree-style summary:

```
📋 Configuration
┌─────────────────────────────────────────────────────────────┐
│ deepseek-r1-1p4d                                            │
└─────────────────────────────────────────────────────────────┘

deepseek-r1-1p4d
├── 📦 Model
│   ├── path: deepseek-r1
│   ├── container: latest
│   └── precision: fp8
├── 🖥️  Resources
│   ├── gpu_type: gb200
│   ├── prefill: 1 workers
│   ├── decode: 4 workers
│   └── gpus_per_node: 4
├── 📊 Benchmark
│   ├── type: sa-bench
│   ├── isl: 1024, osl: 1024
│   └── concurrencies: [128, 256, 512]
└── 🔄 Sweep Parameters (if present)
    ├── chunked_prefill_size: [4096, 8192]
    └── max_total_tokens: [8192, 16384]
```

### Interactive Actions Menu

After viewing the config summary, you'll see an action menu:

```
? What would you like to do?
  🚀 Submit job(s)          - Submit to SLURM cluster
  👁️  Preview sbatch script  - View generated SLURM script with syntax highlighting
  ✏️  Modify parameters      - Interactively change values before submission
  🔍 Dry-run                - Full dry-run preview without submission
  📁 Select different config - Choose a different recipe
  ❌ Exit                   - Exit interactive mode
```

### sbatch Preview

The "Preview sbatch script" option shows the exact SLURM script that will be submitted:

```bash
┌─ Generated sbatch Script ────────────────────────────────────────────────────┐
│  1 │ #!/bin/bash                                                             │
│  2 │ #SBATCH --job-name=deepseek-r1-1p4d                                     │
│  3 │ #SBATCH --nodes=5                                                       │
│  4 │ #SBATCH --gpus-per-node=4                                               │
│  5 │ #SBATCH --time=04:00:00                                                 │
│  6 │ #SBATCH --partition=batch                                               │
│  7 │ ...                                                                     │
└──────────────────────────────────────────────────────────────────────────────┘
```

The script is syntax-highlighted with line numbers for easy reading.

### Parameter Modification

The "Modify parameters" option lets you interactively change key settings:

```
Modify Configuration
Press Enter to keep current value, or type new value

? Job name [deepseek-r1-1p4d]: my-experiment
? Prefill workers [1]:
? Decode workers [4]: 8
? Input sequence length [1024]: 2048
? Output sequence length [1024]: 2048
```

**Modifiable fields:**
- `name` - Job name
- `resources.prefill_workers` - Number of prefill workers
- `resources.decode_workers` - Number of decode workers
- `benchmark.isl` - Input sequence length
- `benchmark.osl` - Output sequence length

Modified configs are saved to a temporary file and used for submission.

### Sweep Preview

For configs with a `sweep:` section, interactive mode shows an expansion table:

```
┌─ Sweep Jobs ────────────────────────────────────────────────────────────────┐
│ #  │ Job Name                           │ Parameters                        │
├────┼────────────────────────────────────┼───────────────────────────────────┤
│ 1  │ deepseek-r1-1p4d_cps4096_mtt8192   │ chunked_prefill_size=4096,        │
│    │                                    │ max_total_tokens=8192              │
│ 2  │ deepseek-r1-1p4d_cps4096_mtt16384  │ chunked_prefill_size=4096,        │
│    │                                    │ max_total_tokens=16384             │
│ 3  │ deepseek-r1-1p4d_cps8192_mtt8192   │ chunked_prefill_size=8192,        │
│    │                                    │ max_total_tokens=8192              │
│ 4  │ deepseek-r1-1p4d_cps8192_mtt16384  │ chunked_prefill_size=8192,        │
│    │                                    │ max_total_tokens=16384             │
└─────────────────────────────────────────────────────────────────────────────┘

Total jobs: 4
```

### Submission Confirmation

Before submitting, you'll be asked to confirm:

```
? Submit to SLURM? (y/N)
```

For sweeps, the confirmation shows:
- Full configuration summary
- Sweep expansion table
- Total job count

### Workflow Examples

**Exploring a new recipe:**
```
$ srtctl
> Select: gb200-fp8/sglang-1p4d.yaml
> Action: 👁️  Preview sbatch script  (review generated script)
> Action: 🔍 Dry-run                 (full dry-run)
> Action: 📁 Select different config (try another)
```

**Quick experiment with modifications:**
```
$ srtctl
> Select: gb200-fp8/sglang-1p4d.yaml
> Action: ✏️  Modify parameters
  > Change decode_workers: 8
  > Change isl: 2048
> Action: 🚀 Submit job(s)
> Confirm: y
```

**Sweep validation:**
```
$ srtctl
> Select: configs/my-sweep.yaml
> View: Sweep table showing 16 jobs
> Action: 🔍 Dry-run (saves all expanded configs to dry-runs/)
> Review generated configs
> Action: 🚀 Submit job(s)
```

## Commands

### `srtctl apply`

Submit a job or sweep to SLURM.

```bash
srtctl apply -f <config.yaml> [options]
```

**Options:**

| Flag | Description |
|------|-------------|
| `-f, --file` | Path to YAML config file (required) |
| `--sweep` | Force sweep mode (usually auto-detected) |
| `--setup-script` | Custom setup script from `configs/` |
| `--tags` | Comma-separated tags for the run |
| `-y, --yes` | Skip confirmation prompts |

**Examples:**

```bash
# Submit single job
srtctl apply -f recipes/gb200-fp8/sglang-1p4d.yaml

# Submit sweep (auto-detected from sweep: section)
srtctl apply -f configs/my-sweep.yaml

# With tags
srtctl apply -f config.yaml --tags "experiment-1,baseline"
```

### `srtctl dry-run`

Preview what would be submitted without actually submitting.

```bash
srtctl dry-run -f <config.yaml> [options]
```

**Options:**

| Flag | Description |
|------|-------------|
| `-f, --file` | Path to YAML config file (required) |
| `--sweep` | Force sweep mode |

**Examples:**

```bash
# Preview single job - shows sbatch script
srtctl dry-run -f config.yaml

# Preview sweep - shows job table and saves configs
srtctl dry-run -f sweep-config.yaml
```

Dry-run output includes:
- Syntax-highlighted sbatch script
- For sweeps: table of all jobs with parameters
- Generated configs saved to `dry-runs/` folder

## Output

When you submit a job, `srtctl` creates an output directory:

```
outputs/<job_id>/
├── config.yaml         # Copy of submitted config
├── sbatch_script.sh    # Generated SLURM script
└── <job_id>.json       # Job metadata
```

## Sweep Support

Configs with a `sweep:` section are automatically detected and expanded:

```yaml
sweep:
  chunked_prefill_size: [4096, 8192]
  max_total_tokens: [8192, 16384]
```

This creates 4 jobs (2 × 2 Cartesian product). See [Parameter Sweeps](sweeps.md) for details.

## Tips

- Use `srtctl` (no args) for exploring recipes interactively
- Use `srtctl apply -f` for scripting and CI pipelines
- Always `dry-run` first for sweeps to check job count
- Check `outputs/<job_id>/` for submitted configs and metadata


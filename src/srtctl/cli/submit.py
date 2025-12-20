#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unified job submission interface for srtctl.

This is the main entrypoint for submitting benchmarks via YAML configs.

Usage:
    srtctl apply -f config.yaml                     # Submit using typed config
    srtctl apply -f config.yaml --use-orchestrator  # Use new Python orchestrator
    srtctl dry-run -f sweep.yaml --sweep
"""

import argparse
import contextlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import yaml

# Import from srtctl modules
from srtctl.core.config import get_srtslurm_setting, load_config
from srtctl.core.schema import SrtConfig


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def calculate_required_nodes(config: SrtConfig) -> int:
    """Calculate total number of nodes required based on resource allocation."""
    return config.resources.total_nodes


def calculate_required_nodes_from_dict(config: dict) -> int:
    """Calculate total number of nodes required from dict config (legacy)."""
    resources = config.get("resources", {})

    prefill_nodes = resources.get("prefill_nodes", 0) or 0
    decode_nodes = resources.get("decode_nodes", 0) or 0
    agg_nodes = resources.get("agg_nodes", 0) or 0

    total_nodes = prefill_nodes + decode_nodes + agg_nodes

    if total_nodes == 0:
        prefill_workers = resources.get("prefill_workers", 0) or 0
        decode_workers = resources.get("decode_workers", 0) or 0
        agg_workers = resources.get("agg_workers", 0) or 0
        total_nodes = prefill_workers + decode_workers + agg_workers

    return max(total_nodes, 1)


def generate_minimal_sbatch_script(
    config: SrtConfig,
    config_path: Path,
) -> str:
    """Generate minimal sbatch script that calls the Python orchestrator.

    The orchestrator runs INSIDE the container on the head node.
    srtctl is pip-installed inside the container at job start.

    Args:
        config: Typed SrtConfig
        config_path: Path to the YAML config file

    Returns:
        Rendered sbatch script as string
    """
    from jinja2 import Environment, FileSystemLoader

    # Find template directory and srtctl source
    srtctl_root = get_srtslurm_setting("srtctl_root")
    if srtctl_root:
        template_dir = Path(srtctl_root) / "scripts" / "templates"
        srtctl_source = Path(srtctl_root)
    else:
        # srtctl source is the parent of src/srtctl (i.e., the repo root)
        srtctl_source = Path(__file__).parent.parent.parent.parent
        template_dir = srtctl_source / "scripts" / "templates"

    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("job_script_minimal.j2")

    total_nodes = calculate_required_nodes(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Resolve container image path (expand aliases from srtslurm.yaml)
    container_image = os.path.expandvars(config.model.container)

    rendered = template.render(
        job_name=config.name,
        total_nodes=total_nodes,
        gpus_per_node=config.resources.gpus_per_node,
        account=config.slurm.account or os.environ.get("SLURM_ACCOUNT", "default"),
        partition=config.slurm.partition or os.environ.get("SLURM_PARTITION", "default"),
        time_limit=config.slurm.time_limit or "01:00:00",
        config_path=str(config_path.resolve()),
        timestamp=timestamp,
        use_gpus_per_node_directive=get_srtslurm_setting("use_gpus_per_node_directive", True),
        use_segment_sbatch_directive=get_srtslurm_setting("use_segment_sbatch_directive", True),
        sbatch_directives=config.sbatch_directives,
        container_image=container_image,
        srtctl_source=str(srtctl_source.resolve()),
    )

    return rendered


def submit_with_orchestrator(
    config_path: Path,
    config: SrtConfig | None = None,
    dry_run: bool = False,
    tags: list[str] | None = None,
) -> None:
    """Submit job using the new Python orchestrator.

    This uses the minimal sbatch template that calls srtctl.cli.do_sweep.

    Args:
        config_path: Path to YAML config file
        config: Pre-loaded SrtConfig (or None to load from path)
        dry_run: If True, print script but don't submit
        tags: Optional tags for the run
    """
    if config is None:
        config = load_config(config_path)

    script_content = generate_minimal_sbatch_script(
        config=config,
        config_path=config_path,
    )

    if dry_run:
        print(f"\n{'=' * 60}")
        print(f"🔍 DRY-RUN (orchestrator mode): {config.name}")
        print(f"{'=' * 60}")
        print("\nGenerated sbatch script:")
        print("-" * 60)
        print(script_content)
        print("-" * 60)
        return

    # Write script to temp file
    fd, script_path = tempfile.mkstemp(suffix=".slurm", prefix="srtctl_", text=True)
    with os.fdopen(fd, "w") as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)

    logging.info(f"🚀 Submitting job (orchestrator mode): {config.name}")
    logging.info(f"Script: {script_path}")

    try:
        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True,
            text=True,
            check=True,
        )

        job_id = result.stdout.strip().split()[-1]
        logging.info(f"✅ Job submitted: {job_id}")

        output_dir = Path("./outputs") / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(config_path, output_dir / "config.yaml")
        shutil.copy(script_path, output_dir / "sbatch_script.sh")

        # Build comprehensive job metadata
        metadata = {
            "version": "2.0",
            "orchestrator": True,
            "job_id": job_id,
            "job_name": config.name,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # Model info
            "model": {
                "path": config.model.path,
                "container": config.model.container,
                "precision": config.model.precision,
            },
            # Resource allocation
            "resources": {
                "gpu_type": config.resources.gpu_type,
                "gpus_per_node": config.resources.gpus_per_node,
                "prefill_nodes": config.resources.prefill_nodes,
                "decode_nodes": config.resources.decode_nodes,
                "prefill_workers": config.resources.num_prefill,
                "decode_workers": config.resources.num_decode,
                "agg_workers": config.resources.num_agg,
            },
            # Backend and frontend
            "backend_type": config.backend_type,
            "use_sglang_router": config.frontend.use_sglang_router,
            # Benchmark config
            "benchmark": {
                "type": config.benchmark.type,
                "isl": config.benchmark.isl,
                "osl": config.benchmark.osl,
            },
        }
        if tags:
            metadata["tags"] = tags
        if config.setup_script:
            metadata["setup_script"] = config.setup_script

        with open(output_dir / f"{job_id}.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✅ Job {job_id} submitted!")
        print(f"📁 Output: {output_dir}")
        print(f"📋 Monitor: tail -f {output_dir}/logs/sweep_{job_id}.log")

    except subprocess.CalledProcessError as e:
        logging.error(f"❌ sbatch failed: {e.stderr}")
        raise
    finally:
        with contextlib.suppress(OSError):
            os.remove(script_path)


def submit_single(
    config_path: Path | None = None,
    config: SrtConfig | None = None,
    dry_run: bool = False,
    setup_script: str | None = None,
    tags: list[str] | None = None,
):
    """Submit a single job from YAML config.

    Uses the orchestrator by default. This is the recommended submission method.

    Args:
        config_path: Path to YAML config file
        config: Pre-loaded SrtConfig (or None if loading from path)
        dry_run: If True, don't submit to SLURM
        setup_script: Optional custom setup script name
        tags: Optional list of tags
    """
    if config is None and config_path:
        config = load_config(config_path)

    if config is None:
        raise ValueError("Either config_path or config must be provided")

    # Always use orchestrator mode
    submit_with_orchestrator(
        config_path=config_path or Path("./config.yaml"),
        config=config,
        dry_run=dry_run,
        tags=tags,
    )


def is_sweep_config(config_path: Path) -> bool:
    """Check if config file is a sweep config by looking for 'sweep' section."""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return "sweep" in config if config else False
    except Exception:
        return False


def submit_sweep(
    config_path: Path,
    dry_run: bool = False,
    setup_script: str | None = None,
    tags: list[str] | None = None,
):
    """Submit parameter sweep.

    Args:
        config_path: Path to sweep YAML config
        dry_run: If True, don't submit to SLURM
        setup_script: Optional custom setup script name
        tags: Optional list of tags
    """
    from srtctl.core.sweep import generate_sweep_configs

    with open(config_path) as f:
        sweep_config = yaml.safe_load(f)

    configs = generate_sweep_configs(sweep_config)
    logging.info(f"Generated {len(configs)} configurations for sweep")

    if dry_run:
        logging.info(f"🔍 DRY-RUN MODE: Sweep with {len(configs)} jobs")

        sweep_dir = Path.cwd() / "dry-runs" / f"{sweep_config['name']}_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        sweep_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"📁 Sweep directory: {sweep_dir}")

        with open(sweep_dir / "sweep_config.yaml", "w") as f:
            yaml.dump(sweep_config, f, default_flow_style=False)

        for i, (config_dict, params) in enumerate(configs, 1):
            job_name = config_dict.get("name", f"job_{i}")
            logging.info(f"\n[{i}/{len(configs)}] {job_name}")
            logging.info(f"  Parameters: {params}")

            job_dir = sweep_dir / f"job_{i:03d}_{job_name}"
            job_dir.mkdir(exist_ok=True)

            with open(job_dir / "config.yaml", "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)

            logging.info(f"  ✓ {job_dir.name}")

        print(
            f"\n{'=' * 60}\n🔍 SWEEP: {sweep_config['name']} ({len(configs)} jobs)\nOutput: {sweep_dir}\n{'=' * 60}\n"
        )
        return

    # Real submission
    for i, (config_dict, params) in enumerate(configs, 1):
        job_name = config_dict.get("name", f"job_{i}")
        logging.info(f"\n[{i}/{len(configs)}] Submitting: {job_name}")
        logging.info(f"  Parameters: {params}")

        # Save temp config and submit
        fd, temp_config_path = tempfile.mkstemp(suffix=".yaml", prefix="srtctl_sweep_", text=True)
        try:
            with os.fdopen(fd, "w") as f:
                yaml.dump(config_dict, f)

            config = load_config(Path(temp_config_path))
            submit_single(
                config_path=Path(temp_config_path),
                config=config,
                dry_run=False,
                setup_script=setup_script,
                tags=tags,
            )
        finally:
            with contextlib.suppress(OSError):
                os.remove(temp_config_path)


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="srtctl - SLURM job submission",
        epilog="""Examples:
  srtctl apply -f config.yaml                    # Submit job (orchestrator mode)
  srtctl apply -f config.yaml --sweep            # Submit sweep
  srtctl dry-run -f config.yaml                  # Dry run
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_args(p):
        p.add_argument("-f", "--file", type=Path, required=True, dest="config", help="YAML config file")
        p.add_argument("--sweep", action="store_true", help="Force sweep mode")

    apply_parser = subparsers.add_parser("apply", help="Submit job(s) to SLURM")
    add_common_args(apply_parser)
    apply_parser.add_argument("--setup-script", type=str, help="Custom setup script in configs/")
    apply_parser.add_argument("--tags", type=str, help="Comma-separated tags")

    dry_run_parser = subparsers.add_parser("dry-run", help="Validate without submitting")
    add_common_args(dry_run_parser)

    args = parser.parse_args()
    if not args.config.exists():
        logging.error(f"Config not found: {args.config}")
        sys.exit(1)

    is_dry_run = args.command == "dry-run"
    is_sweep = args.sweep or is_sweep_config(args.config)
    tags = [t.strip() for t in (getattr(args, "tags", "") or "").split(",") if t.strip()] or None

    try:
        setup_script = getattr(args, "setup_script", None)

        if is_sweep:
            submit_sweep(args.config, dry_run=is_dry_run, setup_script=setup_script, tags=tags)
        else:
            submit_single(config_path=args.config, dry_run=is_dry_run, setup_script=setup_script, tags=tags)
    except Exception as e:
        logging.exception(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

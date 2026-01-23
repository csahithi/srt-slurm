#!/usr/bin/env python3
"""Upload rollup.json files to SRT endpoint as Parquet files grouped by backend/frontend/benchmarker.

Process:
1. Find all rollup.json files in the directory
2. Create flattened JSON for each concurrency level
3. Combine all into a single Parquet file
4. Group by (backend_type, frontend_type, benchmark_type)
5. Upload separate Parquet file for each group

Usage:
    python upload_rollup.py <directory> <user_login> [--study-id STUDY] [--endpoint URL]

Example:
    python upload_rollup.py ./outputs user@example.com
    python upload_rollup.py ./outputs user@example.com --study-id my-study
    python upload_rollup.py ./outputs user@example.com --endpoint that accepts the upload
"""

import argparse
from collections import defaultdict
import json
import sys
from pathlib import Path

import pandas as pd
import requests


DEFAULT_WORKDIR = Path("/tmp/srt")


def _is_numeric_list(obj: list) -> bool:
    """Check if a list contains only ints or floats."""
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in obj)


def _is_cuda_graph_key(key: str) -> bool:
    """Check if a key is related to cuda graph config."""
    key_lower = key.lower()
    return "cuda_graph" in key_lower or "cudagraph" in key_lower


def flatten_json(obj: dict | list, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested JSON object using dot notation.

    Args:
        obj: The object to flatten (dict or list)
        parent_key: The parent key prefix
        sep: Separator to use between keys (default: ".")

    Returns:
        Flattened dictionary with dot-notation keys

    Note:
        Lists of ints/floats with "cuda_graph" or "cudagraph" in the key are kept as lists.
    """
    items = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, list):
                # Keep numeric lists for cuda_graph configs as-is
                if _is_cuda_graph_key(new_key) and _is_numeric_list(value):
                    items[new_key] = value
                else:
                    items.update(flatten_json(value, new_key, sep))
            elif isinstance(value, dict):
                items.update(flatten_json(value, new_key, sep))
            else:
                items[new_key] = value
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            new_key = f"{parent_key}{sep}{idx}" if parent_key else str(idx)
            if isinstance(value, (dict, list)):
                items.update(flatten_json(value, new_key, sep))
            else:
                items[new_key] = value

    return items


def parse_concurrencies(concurrencies_str: str) -> list[int]:
    """Parse concurrencies string delimited by 'x'.

    Args:
        concurrencies_str: String like "32x64x128" or "48"

    Returns:
        List of integer concurrency values
    """
    if not concurrencies_str:
        return []

    parts = concurrencies_str.split("x")
    result = []
    for part in parts:
        part = part.strip()
        if part:
            try:
                result.append(int(part))
            except ValueError:
                pass
    return result


def extract_launch_commands(data: dict) -> dict:
    """Extract launch commands from all workers, prefixed by node_name.

    Args:
        data: The rollup.json data

    Returns:
        Flattened dict with keys like "launch_commands.prefill"
    """
    nodes_summary = data.get("nodes_summary", {})
    nodes = nodes_summary.get("nodes", [])

    node_type_to_launch_command = defaultdict(list)

    for node in nodes:
        node_name = node.get("node_name", "unknown")
        worker_type = node.get("worker_type", "unknown")
        launch_command = node.get("launch_command", {})
        raw_command = launch_command.get("raw_command", "")

        worker_env_config = data.get("environment_config", {}).get(f"{worker_type}_environment", {})
        engine_config = data.get("environment_config", {}).get(f"{worker_type}_engine_config", {})

        raw_command_str = f"#{worker_type}:{node_name}"
        if worker_env_config:
            raw_command_str += f"\n#env_vars:{worker_env_config}"
        if engine_config:
            raw_command_str += f"\n#engine_config:{engine_config}"
        raw_command_str += f"\n{raw_command}"

        node_type_to_launch_command[worker_type].append(raw_command)

    return node_type_to_launch_command


def create_per_concurrency_rows(data: dict) -> list[dict]:
    """Create one flattened row dict per concurrency level.

    Args:
        data: The rollup.json data

    Returns:
        List of flattened row dicts, one per concurrency
    """
    concurrencies_str = data.get("concurrencies", "")
    concurrencies = parse_concurrencies(concurrencies_str)
    results = data.get("results", [])

    # Build a map of concurrency -> result
    result_by_concurrency = {}
    for result in results:
        conc = result.get("concurrency")
        if conc is not None:
            result_by_concurrency[conc] = result

    # If no concurrencies parsed, try to get from results
    if not concurrencies:
        concurrencies = list(result_by_concurrency.keys())

    # Extract top-level metadata (exclude results, nodes_summary arrays)
    top_level_keys = [
        "job_id", "job_name", "generated_at", "model_path", "model_name",
        "precision", "gpu_type", "gpus_per_node", "backend_type", "frontend_type",
        "is_disaggregated", "total_nodes", "total_gpus", "prefill_nodes",
        "decode_nodes", "prefill_workers", "decode_workers", "prefill_gpus",
        "decode_gpus", "agg_nodes", "agg_workers", "benchmark_type", "isl", "osl",
        "concurrencies"
    ]

    top_level = {k: data.get(k) for k in top_level_keys if k in data}

    # Extract nodes_summary scalars (not the nodes array)
    nodes_summary = data.get("nodes_summary", {})
    for key, value in nodes_summary.items():
        if key != "nodes" and not isinstance(value, (list, dict)):
            top_level[f"nodes_summary.{key}"] = value

    # Extract environment_config flattened
    env_config = data.get("environment_config", {})
    if env_config:
        flattened_env = flatten_json(env_config, "environment_config")
        top_level.update(flattened_env)

    # Extract benchmark_command raw_command only
    benchmark_cmd = data.get("benchmark_command", {})
    if benchmark_cmd:
        raw_cmd = benchmark_cmd.get("raw_command")
        
        if raw_cmd:
            top_level["benchmark_command"] = raw_cmd
    
    sbatch_script = data.get("sbatch_script")
    if sbatch_script:
        top_level.update(
            {
                "benchmark_command": f"#sbatch_script\n\n{raw_cmd}",
            }
        )

    # Extract launch commands from all workers
    launch_commands = extract_launch_commands(data)
    for worker_type, commands in launch_commands.items():
        top_level[f"launch_commands.{worker_type}"] = "\n".join(commands)

    rows = []

    for concurrency in concurrencies:
        row = dict(top_level)
        row["concurrency"] = concurrency

        # Add result for this concurrency with "result_" prefix
        result = result_by_concurrency.get(concurrency, {})
        for key, value in result.items():
            row[f"result_{key}"] = value

        rows.append(row)

    return rows


def upload_parquet(
    parquet_path: Path,
    user_login: str,
    session_id: str,
    endpoint: str,
    backend: str,
    benchmarker: str,
    frontend: str,
    mode: str,
) -> tuple[bool, str]:
    """Upload a Parquet file to the endpoint.

    Args:
        parquet_path: Path to the Parquet file
        user_login: User login/email
        session_id: Session ID for the upload
        endpoint: API endpoint URL
        backend: Backend type
        benchmarker: Benchmark type
        frontend: Frontend type
        mode: Mode (disaggregated or aggregated)

    Returns:
        Tuple of (success, message)
    """
    parquet_content = parquet_path.read_bytes()

    try:
        response = requests.post(
            endpoint,
            files={"file": (parquet_path.name, parquet_content, "application/octet-stream")},
            data={
                "user_login": user_login,
                "session_id": session_id,
                "backend": backend,
                "benchmarker": benchmarker,
                "frontend": frontend,
                "mode": mode,
            },
            timeout=60,
        )

        if response.ok:
            return True, f"HTTP {response.status_code}"
        else:
            return False, f"HTTP {response.status_code}: {response.text}"

    except requests.RequestException as e:
        return False, f"Request failed: {e}"


def find_rollup_files(directory: Path) -> list[Path]:
    """Recursively find all rollup.json files in a directory."""
    return list(directory.rglob("rollup.json"))


def read_sbatch_script(rollup_path: Path) -> str | None:
    """Read the sbatch_script.sh associated with a rollup.json.

    The sbatch script is expected to be at <job_dir>/sbatch_script.sh
    where rollup.json is at <job_dir>/logs/rollup.json.

    Args:
        rollup_path: Path to the rollup.json file

    Returns:
        Content of sbatch_script.sh or None if not found
    """
    # rollup.json is at <job_dir>/logs/rollup.json
    # sbatch_script.sh is at <job_dir>/sbatch_script.sh
    job_dir = rollup_path.parent.parent
    sbatch_path = job_dir / "sbatch_script.sh"

    if sbatch_path.exists():
        try:
            return sbatch_path.read_text()
        except Exception:
            return None
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Upload rollup.json files to SRT endpoint as Parquet (grouped by backend/frontend/benchmarker)"
    )
    parser.add_argument("directory", type=Path, help="Directory to search for rollup.json files")
    parser.add_argument("user_login", help="User login/email for the upload")
    parser.add_argument(
        "--study-id",
        help="Study ID (default: extracted from first job_name per group)",
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000",
        help="API endpoint (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        help=f"Working directory for output files (default: {DEFAULT_WORKDIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate Parquet files but don't upload",
    )
    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Keep generated Parquet files after upload (default: delete)",
    )

    args = parser.parse_args()

    if not args.directory.exists():
        print(f"Error: Directory does not exist: {args.directory}", file=sys.stderr)
        sys.exit(1)

    # Setup workdir
    workdir = args.workdir if args.workdir else DEFAULT_WORKDIR
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"Working directory: {workdir}")

    rollup_files = find_rollup_files(args.directory)

    if not rollup_files:
        print(f"No rollup.json files found in {args.directory}")
        sys.exit(0)

    print(f"Found {len(rollup_files)} rollup.json files")
    print(f"User: {args.user_login}")
    print(f"Endpoint: {args.endpoint}")
    print(f"Study ID: {args.study_id}")
    print("---")

    all_rows = []
    failed_count = 0

    for rollup_path in sorted(rollup_files):
        print(f"Processing: {rollup_path}")

        try:
            with open(rollup_path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ✗ Failed to read: {e}")
            failed_count += 1
            continue

        # Read sbatch script for this job
        sbatch_script = read_sbatch_script(rollup_path)

        # Create per-concurrency rows
        rows = create_per_concurrency_rows(data)

        if not rows:
            print("  ⚠ No concurrency results found")
            continue

        # Add sbatch script to each row
        for row in rows:
            if sbatch_script:
                row["sbatch_script"] = sbatch_script
            print(f"  Added: job_id={row.get('job_id')}, concurrency={row.get('concurrency')}")

        all_rows.extend(rows)

    print("---")
    print(f"Total rollups processed: {len(rollup_files)}")
    print(f"Total rows: {len(all_rows)}")
    print(f"Failed to read: {failed_count}")

    if not all_rows:
        print("No data to write")
        sys.exit(0)

    # Create full DataFrame
    df = pd.DataFrame(all_rows)
    # add metadata columns
    df["experiment_id"] = 'STUDY'
    df["study_id"] = args.study_id
    df["user_login"] = args.user_login
    
    # Write combined Parquet
    combined_parquet = workdir / "rollup_results_combined.parquet"
    df.to_parquet(combined_parquet, index=False)
    print(f"\n✓ Created combined Parquet: {combined_parquet}")
    print(f"  Total rows: {len(df)}")
    print(f"  Total columns: {len(df.columns)}")

    # Group by backend_type, frontend_type, benchmark_type, is_disaggregated
    group_cols = ["backend_type", "frontend_type", "benchmark_type", "is_disaggregated"]

    # Fill NaN with 'unknown' for grouping (handle is_disaggregated specially)
    for col in group_cols:
        if col not in df.columns:
            df[col] = "unknown" if col != "is_disaggregated" else False
        elif col == "is_disaggregated":
            df[col] = df[col].fillna(False)
        else:
            df[col] = df[col].fillna("unknown")

    grouped = df.groupby(group_cols)
    print(f"\nFound {len(grouped)} unique groups:")

    generated_files = [combined_parquet]
    success_count = 0
    upload_failed_count = 0

    for (backend, frontend, benchmarker, is_disagg), group_df in grouped:
        mode = "disaggregated" if is_disagg else "aggregated"
        print(f"\n--- Group: backend={backend}, frontend={frontend}, benchmarker={benchmarker}, mode={mode} ---")
        print(f"  Rows: {len(group_df)}")

        # Create group Parquet filename
        safe_backend = str(backend).replace("/", "_")
        safe_frontend = str(frontend).replace("/", "_")
        safe_benchmarker = str(benchmarker).replace("/", "_")
        group_filename = f"rollup_{safe_backend}_{safe_frontend}_{safe_benchmarker}_{mode}.parquet"
        group_parquet = workdir / group_filename

        # Write group Parquet
        group_df.to_parquet(group_parquet, index=False)
        generated_files.append(group_parquet)
        print(f"  ✓ Created: {group_parquet}")

        if args.dry_run:
            print("  Dry run - skipping upload")
            continue

        # Determine study_id - use provided or first job_name in group
        if args.study_id:
            study_id = args.study_id
        else:
            study_id = group_df["job_name"].iloc[0] if "job_name" in group_df.columns else "unknown"

        print(f"  Uploading with study_id: {study_id}")

        success, message = upload_parquet(
            group_parquet,
            args.user_login,
            study_id,
            args.endpoint,
            backend=str(backend),
            benchmarker=str(benchmarker),
            frontend=str(frontend),
            mode=mode,
        )

        if success:
            print(f"  ✓ Uploaded ({message})")
            success_count += 1
        else:
            print(f"  ✗ Upload failed: {message}")
            upload_failed_count += 1

    # Cleanup unless --keep-files
    if not args.keep_files and not args.dry_run:
        for fp in generated_files:
            try:
                fp.unlink()
            except Exception:
                pass
        print(f"\nCleaned up {len(generated_files)} generated files")

    print("\n" + "=" * 50)
    print(f"Total groups: {len(grouped)}")
    if not args.dry_run:
        print(f"Successful uploads: {success_count}")
        print(f"Failed uploads: {upload_failed_count}")

    if upload_failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

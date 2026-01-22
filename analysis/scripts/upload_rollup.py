#!/usr/bin/env python3
"""Upload rollup.json files to SRT endpoint with flattened JSON structure.

For each concurrency level in the rollup, creates a separate flattened JSON with:
- Top-level metadata (duplicated for each concurrency)
- Results for that concurrency prefixed with "result_"
- Launch commands from all workers prefixed with node_name

Usage:
    python upload_rollups.py <directory> <user_login> [--session-id SESSION] [--endpoint URL]

Example:
    python upload_rollups.py ./outputs user@example.com
    python upload_rollups.py ./outputs user@example.com --session-id my-session
    python upload_rollups.py ./outputs user@example.com --workdir /tmp/srt/my-run
"""

import argparse
from collections import defaultdict
import json
import sys
from pathlib import Path

import requests


DEFAULT_WORKDIR = Path("/tmp/srt")


def flatten_json(obj: dict | list, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested JSON object using dot notation.

    Args:
        obj: The object to flatten (dict or list)
        parent_key: The parent key prefix
        sep: Separator to use between keys (default: ".")

    Returns:
        Flattened dictionary with dot-notation keys
    """
    items = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, (dict, list)):
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
        Flattened dict with keys like "worker-0.launch_command.raw_command"
    """
    nodes_summary = data.get("nodes_summary", {})
    nodes = nodes_summary.get("nodes", [])
    
    node_type_to_launch_command = defaultdict(list)

    for node in nodes:
        node_name = node.get("node_name", "unknown")
        worker_type = node.get("worker_type", "unknown")
        launch_command = node.get("launch_command", {})
        raw_command = launch_command.get("raw_command", "")

        raw_command = f"#{worker_type}:{node_name}\n{raw_command}"

        node_type_to_launch_command[worker_type].append(raw_command)


    return node_type_to_launch_command


def create_per_concurrency_jsons(data: dict, workdir: Path) -> list[tuple[int, Path]]:
    """Create one JSON file per concurrency level.

    Args:
        data: The rollup.json data
        workdir: Directory to write output files

    Returns:
        List of (concurrency, file_path) tuples
    """
    job_id = data.get("job_id", "unknown")
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

    # Extract launch commands from all workers
    launch_commands = extract_launch_commands(data)
    for worker_type, commands in launch_commands.items():
        top_level[f"launch_commands.{worker_type}"] = "\n".join(commands)

    output_files = []

    for concurrency in concurrencies:
        output = dict(top_level)
        output["concurrency"] = concurrency

        # Add launch commands
        output.update(launch_commands)

        # Add result for this concurrency with "result_" prefix
        result = result_by_concurrency.get(concurrency, {})
        for key, value in result.items():
            output[f"result_{key}"] = value

        # Write to file
        filename = f"{job_id}_concurrency_{concurrency}.json"
        filepath = workdir / filename

        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)

        output_files.append((concurrency, filepath))

    return output_files


def upload_json(
    json_path: Path,
    user_login: str,
    session_id: str | None,
    endpoint: str,
) -> tuple[bool, str]:
    """Upload a single JSON file to the endpoint.

    Args:
        json_path: Path to the JSON file
        user_login: User login/email
        session_id: Optional session ID (uses job_name if not provided)
        endpoint: API endpoint URL

    Returns:
        Tuple of (success, message)
    """
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception as e:
        return False, f"Failed to read JSON: {e}"

    # Extract metadata
    benchmark_type = data.get("benchmark_type", "unknown")
    frontend_type = data.get("frontend_type", "unknown")
    backend_type = data.get("backend_type", "unknown")
    job_name = data.get("job_name", "unknown")

    # Use provided session_id or fall back to job_name
    effective_session_id = session_id if session_id else job_name

    # Read the file content for upload
    json_content = json_path.read_text()

    try:
        response = requests.post(
            f"{endpoint}/upload/srt",
            files={"file": (json_path.name, json_content, "application/json")},
            data={
                "user_login": user_login,
                "session_id": effective_session_id,
                "backend": backend_type,
                "benchmarker": benchmark_type,
                "frontend": frontend_type,
            },
            timeout=30,
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


def main():
    parser = argparse.ArgumentParser(
        description="Upload rollup.json files to SRT endpoint with flattened JSON"
    )
    parser.add_argument("directory", type=Path, help="Directory to search for rollup.json files")
    parser.add_argument("user_login", help="User login/email for the upload")
    parser.add_argument(
        "--session-id",
        help="Session ID (default: extracted from job_name in each JSON)",
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
        help="Generate files but don't upload",
    )
    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Keep generated files after upload (default: delete)",
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
    if args.session_id:
        print(f"Session ID: {args.session_id}")
    else:
        print("Session ID: (from job_name)")
    print("---")

    success_count = 0
    failed_count = 0
    generated_files = []

    for rollup_path in sorted(rollup_files):
        print(f"Processing: {rollup_path}")

        try:
            with open(rollup_path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ✗ Failed to read: {e}")
            failed_count += 1
            continue

        # Create per-concurrency JSON files
        output_files = create_per_concurrency_jsons(data, workdir)
        generated_files.extend([fp for _, fp in output_files])

        if not output_files:
            print("  ⚠ No concurrency results found")
            continue

        for concurrency, json_path in output_files:
            print(f"  Generated: {json_path.name} (concurrency={concurrency})")

            if args.dry_run:
                continue

            success, message = upload_json(
                json_path,
                args.user_login,
                args.session_id,
                args.endpoint,
            )

            if success:
                print(f"    ✓ Uploaded ({message})")
                success_count += 1
            else:
                print(f"    ✗ Upload failed: {message}")
                failed_count += 1

        print()

    # Cleanup generated files unless --keep-files
    if not args.keep_files and not args.dry_run:
        for fp in generated_files:
            try:
                fp.unlink()
            except Exception:
                pass
        print(f"Cleaned up {len(generated_files)} generated files")

    print("---")
    print(f"Total rollups: {len(rollup_files)}")
    print(f"Generated files: {len(generated_files)}")
    if not args.dry_run:
        print(f"Successful uploads: {success_count}")
        print(f"Failed uploads: {failed_count}")

    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

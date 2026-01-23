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
import gzip
import json
import sys
from pathlib import Path

import requests


DEFAULT_WORKDIR = Path("/tmp/srt")


def upload_json(
    json_path: Path,
    user_login: str,
    session_id: str,
    endpoint: str,
    backend: str,
    benchmarker: str,
    frontend: str,
    mode: str,
) -> tuple[bool, str]:
    """Upload a gzipped JSON file to the endpoint.

    Args:
        json_path: Path to the JSON file
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
    json_content = json_path.read_bytes()
    compressed_content = gzip.compress(json_content)

    # Use .json.gz extension to indicate gzipped JSON
    filename = json_path.name + ".gz"

    try:
        response = requests.post(
            endpoint,
            files={"file": (filename, compressed_content, "application/gzip")},
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

    failed_count = 0

    groups: defaultdict[tuple[str, str, str], list[dict]] = defaultdict(list)

    for rollup_path in sorted(rollup_files):
        print(f"Processing: {rollup_path}")

        try:
            with open(rollup_path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ✗ Failed to read: {e}")
            failed_count += 1
            continue

        # Skip if no nodes_summary (job likely failed/cancelled)
        if not data.get("nodes_summary"):
            print("  ⚠ Skipping: no nodes_summary (job may have failed)")
            continue

        mode = "aggregated" if data.get("is_aggregated") else "disaggregated"
        group = (data['benchmark_type'], data['frontend_type'], data['backend_type'], mode)
        # Read sbatch script for this job
        sbatch_script = read_sbatch_script(rollup_path)

        # Add sbatch script to each row
        if sbatch_script:
            data["sbatch_script"] = sbatch_script
        groups[group].append(data)

    print("---")
    print(f"Total rollups processed: {len(rollup_files)}")
    print(f"Total groups: {len(groups)}")
    for group, rows in groups.items():
        print(f"  {group}: {len(rows)}")
    print(f"Failed to read: {failed_count}")

    if not groups:
        print("No data to write")
        return

    success_count = 0
    upload_failed_count = 0

    for group, rows in groups.items():
        print(f"\n--- Group: {group} ---")
        print(f"  Rows: {len(rows)}")

        group_str = "_".join(group)

        group_filename = f"rollup_{group_str}.json"
        group_path = workdir / group_filename
        with open(group_path, "w") as f:
            json.dump(rows, f, indent=1)
        print(f"  ✓ Created: {group_path}")

        if args.dry_run:
            print("  Dry run - skipping upload")
            continue

        # Determine study_id - use provided or first job_name in group
        if args.study_id:
            study_id = args.study_id
        else:
            study_id = rows[0]["job_name"]

        print(f"  Uploading with study_id: {study_id}")

        success, message = upload_json(
            group_path,
            args.user_login,
            study_id,
            args.endpoint,
            benchmarker=group[0],
            frontend=group[1],
            backend=group[2],
            mode=group[3],
        )

        if success:
            print(f"  ✓ Uploaded ({message})")
            success_count += 1
        else:
            print(f"  ✗ Upload failed: {message}")
            upload_failed_count += 1

    # Cleanup unless --keep-files
    if not args.keep_files and not args.dry_run:
        for fp in workdir.glob("*.json"):
            try:
                fp.unlink()
            except Exception:
                print(f"  ✗ Failed to delete: {fp}")
        print(f"\nCleaned up {len(list(workdir.glob('*.json')))} generated files")

    print("\n" + "=" * 50)
    print(f"Total groups: {len(groups)}")
    if not args.dry_run:
        print(f"Successful uploads: {success_count}")
        print(f"Failed uploads: {upload_failed_count}")

    if upload_failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

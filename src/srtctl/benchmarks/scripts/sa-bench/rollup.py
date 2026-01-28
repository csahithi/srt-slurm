#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate benchmark-rollup.json from sa-bench results."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _get_percentile(percentiles: list, target: float) -> float | None:
    """Extract a specific percentile value from the percentiles list."""
    if not percentiles:
        return None
    for p, v in percentiles:
        if p == target:
            return v
    return None


def main(log_dir: Path) -> None:
    """Generate benchmark-rollup.json from sa-bench result files."""
    result_files = sorted(log_dir.glob("sa-bench_*/results_*.json"))
    if not result_files:
        print("No sa-bench results found", file=sys.stderr)
        return

    runs = []
    config = {}

    for f in result_files:
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            print(f"Failed to parse {f}: {e}", file=sys.stderr)
            continue

        # Extract config from first file
        if not config:
            config = {
                "model": data.get("model_id"),
                "isl": data.get("random_input_len"),
                "osl": data.get("random_output_len"),
            }

        runs.append({
            "concurrency": data.get("max_concurrency"),
            "throughput_toks": data.get("output_throughput"),
            "request_throughput": data.get("request_throughput"),
            "ttft_mean_ms": data.get("mean_ttft_ms"),
            "ttft_p99_ms": _get_percentile(data.get("percentiles_ttft_ms", []), 99.0),
            "tpot_mean_ms": data.get("mean_tpot_ms"),
            "tpot_p99_ms": _get_percentile(data.get("percentiles_tpot_ms", []), 99.0),
            "itl_mean_ms": data.get("mean_itl_ms"),
            "itl_p99_ms": _get_percentile(data.get("percentiles_itl_ms", []), 99.0),
            "e2el_mean_ms": data.get("mean_e2el_ms"),
            "completed_requests": data.get("completed"),
            "total_input_tokens": data.get("total_input"),
            "total_output_tokens": data.get("total_output"),
        })

    rollup = {
        "benchmark_type": "sa-bench",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": config,
        "runs": runs,
    }

    output_path = log_dir / "benchmark-rollup.json"
    output_path.write_text(json.dumps(rollup, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    log_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/logs")
    main(log_dir)

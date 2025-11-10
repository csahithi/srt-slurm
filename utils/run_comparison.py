"""
Run comparison utilities for isolation mode

Provides functions to compare two benchmark runs in detail:
- Configuration flag differences
- Performance metric deltas
- Visual comparison helpers
"""

import pandas as pd


def categorize_flag(flag_name: str) -> str:
    """Categorize a configuration flag for organized display.

    Returns category name like "Parallelism", "Attention", etc.
    """
    flag_lower = flag_name.lower()

    if any(x in flag_lower for x in ["tp", "dp", "pp", "ep", "parallel", "nnodes", "node_rank"]):
        return "Parallelism"
    elif any(x in flag_lower for x in ["attention", "backend", "sampling", "moe"]):
        return "Attention & Backend"
    elif any(
        x in flag_lower
        for x in ["kv_cache", "memory", "mem_", "max_total_tokens", "chunked_prefill"]
    ):
        return "Memory & Cache"
    elif any(x in flag_lower for x in ["schedule", "max_running", "priority"]):
        return "Scheduling"
    elif any(x in flag_lower for x in ["enable_", "disable_", "cuda_graph", "radix"]):
        return "Optimization"
    elif any(x in flag_lower for x in ["disaggregation", "transfer", "decode_", "prefill_"]):
        return "Disaggregation"
    elif any(x in flag_lower for x in ["model", "context_length", "served_model"]):
        return "Model"
    else:
        return "Other"


def compare_configs(config_a: dict, config_b: dict) -> dict:
    """Compare configuration flags between two runs.

    Args:
        config_a: Configuration dict from run A (from parsed config JSON)
        config_b: Configuration dict from run B

    Returns:
        {
            'topology_summary': {...},  # Key topology differences
            'flag_differences': [...],   # List of changed flags
            'identical_flags': [...],    # List of unchanged flags
            'num_differences': int
        }
    """
    # Extract server_args from config structure
    server_args_a = config_a.get("config", {}).get("server_args", {})
    server_args_b = config_b.get("config", {}).get("server_args", {})

    # Compare topology summary
    topology_summary = {
        "prefill_tp": (server_args_a.get("tp_size"), server_args_b.get("tp_size")),
        "prefill_dp": (server_args_a.get("dp_size"), server_args_b.get("dp_size")),
        "model": (server_args_a.get("served_model_name"), server_args_b.get("served_model_name")),
        "context_length": (
            server_args_a.get("context_length"),
            server_args_b.get("context_length"),
        ),
    }

    # Get all unique flags from both runs
    all_flags = set(server_args_a.keys()) | set(server_args_b.keys())

    flag_differences = []
    identical_flags = []

    for flag in sorted(all_flags):
        value_a = server_args_a.get(flag)
        value_b = server_args_b.get(flag)

        if value_a != value_b:
            flag_differences.append(
                {
                    "flag": flag,
                    "category": categorize_flag(flag),
                    "run_a_value": value_a if value_a is not None else "Not set",
                    "run_b_value": value_b if value_b is not None else "Not set",
                }
            )
        else:
            identical_flags.append({"flag": flag, "value": value_a})

    # Sort differences by category, then by flag name
    flag_differences.sort(key=lambda x: (x["category"], x["flag"]))

    return {
        "topology_summary": topology_summary,
        "flag_differences": flag_differences,
        "identical_flags": identical_flags,
        "num_differences": len(flag_differences),
    }


def compare_metrics(run_a: dict, run_b: dict) -> pd.DataFrame:
    """Compare performance metrics between two runs at matching concurrency levels.

    Args:
        run_a: Run data dict with metrics arrays
        run_b: Run data dict with metrics arrays

    Returns:
        DataFrame with columns: Concurrency, Metric, Run A, Run B, Delta, % Change
    """

    # Calculate derived metrics if not already present
    def ensure_derived_metrics(run: dict) -> dict:
        """Ensure run has output_tps_per_gpu and output_tps_per_user calculated."""
        if "output_tps_per_gpu" not in run or "output_tps_per_user" not in run:
            # Calculate total GPUs
            total_gpus = run.get("prefill_tp", 0) * run.get("prefill_dp", 0) + run.get(
                "decode_tp", 0
            ) * run.get("decode_dp", 0)
            if total_gpus == 0:
                total_gpus = 1  # Avoid division by zero

            # Calculate Output TPS/GPU
            output_tps = run.get("output_tps", [])
            run["output_tps_per_gpu"] = [tps / total_gpus for tps in output_tps]

            # Calculate Output TPS/User as 1000 / TPOT
            mean_tpot = run.get("mean_tpot_ms", [])
            run["output_tps_per_user"] = [1000 / tpot if tpot > 0 else 0 for tpot in mean_tpot]

        return run

    run_a = ensure_derived_metrics(run_a)
    run_b = ensure_derived_metrics(run_b)

    # Extract metrics that we want to compare
    metrics_to_compare = [
        ("Output TPS", "output_tps"),
        ("Output TPS/GPU", "output_tps_per_gpu"),
        ("Output TPS/User", "output_tps_per_user"),
        ("Mean TTFT (ms)", "mean_ttft_ms"),
        ("P99 TTFT (ms)", "p99_ttft_ms"),
        ("Mean TPOT (ms)", "mean_tpot_ms"),
        ("P99 TPOT (ms)", "p99_tpot_ms"),
        ("Mean ITL (ms)", "mean_itl_ms"),
        ("P99 ITL (ms)", "p99_itl_ms"),
        ("Mean E2EL (ms)", "mean_e2el_ms"),
    ]

    concurrencies_a = run_a.get("concurrencies", [])
    concurrencies_b = run_b.get("concurrencies", [])

    # Find common concurrency levels
    common_concurrencies = set(concurrencies_a) & set(concurrencies_b)

    if not common_concurrencies:
        # No matching concurrency levels
        return pd.DataFrame()

    comparison_rows = []

    for concurrency in sorted(common_concurrencies):
        idx_a = concurrencies_a.index(concurrency)
        idx_b = concurrencies_b.index(concurrency)

        for metric_name, metric_key in metrics_to_compare:
            values_a = run_a.get(metric_key, [])
            values_b = run_b.get(metric_key, [])

            if idx_a < len(values_a) and idx_b < len(values_b):
                value_a = values_a[idx_a]
                value_b = values_b[idx_b]

                if value_a and value_b and value_a != 0:
                    delta = value_b - value_a
                    pct_change = (delta / value_a) * 100

                    # Determine if this is an improvement
                    # For latency metrics (lower is better), negative delta is good
                    # For throughput metrics (higher is better), positive delta is good
                    is_latency = "ms" in metric_name
                    is_improvement = (delta < 0) if is_latency else (delta > 0)

                    comparison_rows.append(
                        {
                            "Concurrency": concurrency,
                            "Metric": metric_name,
                            "Run A": value_a,
                            "Run B": value_b,
                            "Delta": delta,
                            "% Change": pct_change,
                            "Improved": is_improvement,
                        }
                    )

    return pd.DataFrame(comparison_rows)


def calculate_summary_scorecard(comparison_df: pd.DataFrame) -> dict:
    """Calculate summary statistics for the comparison.

    Args:
        comparison_df: DataFrame from compare_metrics()

    Returns:
        {
            'num_improved': int,
            'num_regressed': int,
            'num_unchanged': int,
            'avg_improvement_pct': float,
            'biggest_improvement': str,
            'biggest_regression': str
        }
    """
    if comparison_df.empty:
        return {
            "num_improved": 0,
            "num_regressed": 0,
            "num_unchanged": 0,
            "avg_improvement_pct": 0.0,
            "biggest_improvement": None,
            "biggest_regression": None,
        }

    # Count improvements/regressions across all concurrency levels
    # Group by metric to avoid counting same metric multiple times
    metric_summary = comparison_df.groupby("Metric").agg(
        {
            "Improved": "mean",  # If mostly improved across concurrencies
            "% Change": "mean",
        }
    )

    num_improved = (metric_summary["Improved"] > 0.5).sum()
    num_regressed = (metric_summary["Improved"] < 0.5).sum()
    num_unchanged = (metric_summary["Improved"] == 0.5).sum()

    # Find biggest changes
    abs_changes = comparison_df.copy()
    abs_changes["Abs % Change"] = abs_changes["% Change"].abs()

    if not abs_changes.empty:
        biggest_improvement_row = abs_changes[abs_changes["Improved"]].nlargest(
            1, "Abs % Change", keep="first"
        )
        biggest_regression_row = abs_changes[not abs_changes["Improved"]].nlargest(
            1, "Abs % Change", keep="first"
        )

        biggest_improvement = None
        biggest_regression = None

        if not biggest_improvement_row.empty:
            row = biggest_improvement_row.iloc[0]
            biggest_improvement = f"{row['Metric']}: {row['% Change']:.1f}% improvement"

        if not biggest_regression_row.empty:
            row = biggest_regression_row.iloc[0]
            biggest_regression = f"{row['Metric']}: {abs(row['% Change']):.1f}% regression"
    else:
        biggest_improvement = None
        biggest_regression = None

    # Average improvement percentage (across improved metrics only)
    improved_metrics = comparison_df[comparison_df["Improved"]]
    avg_improvement_pct = improved_metrics["% Change"].mean() if not improved_metrics.empty else 0.0

    return {
        "num_improved": int(num_improved),
        "num_regressed": int(num_regressed),
        "num_unchanged": int(num_unchanged),
        "avg_improvement_pct": float(avg_improvement_pct),
        "biggest_improvement": biggest_improvement,
        "biggest_regression": biggest_regression,
    }


def get_delta_data_for_graphs(run_a: dict, run_b: dict) -> pd.DataFrame:
    """Prepare delta data for visualization graphs.

    Args:
        run_a: Run data dict
        run_b: Run data dict

    Returns:
        DataFrame with columns: Concurrency, TTFT Delta, TPOT Delta, ITL Delta, Throughput Delta
    """
    concurrencies_a = run_a.get("concurrencies", [])
    concurrencies_b = run_b.get("concurrencies", [])

    common_concurrencies = set(concurrencies_a) & set(concurrencies_b)

    delta_rows = []

    for concurrency in sorted(common_concurrencies):
        idx_a = concurrencies_a.index(concurrency)
        idx_b = concurrencies_b.index(concurrency)

        row = {"Concurrency": concurrency}

        # Calculate deltas for key metrics
        metrics = {
            "TTFT Delta (ms)": ("mean_ttft_ms", idx_a, idx_b),
            "TPOT Delta (ms)": ("mean_tpot_ms", idx_a, idx_b),
            "ITL Delta (ms)": ("mean_itl_ms", idx_a, idx_b),
            "Throughput Delta (TPS)": ("output_tps", idx_a, idx_b),
        }

        for delta_name, (metric_key, idx_a, idx_b) in metrics.items():
            values_a = run_a.get(metric_key, [])
            values_b = run_b.get(metric_key, [])

            if idx_a < len(values_a) and idx_b < len(values_b):
                value_a = values_a[idx_a]
                value_b = values_b[idx_b]

                if value_a and value_b:
                    row[delta_name] = value_b - value_a

        if len(row) > 1:  # Has at least concurrency + one metric
            delta_rows.append(row)

    return pd.DataFrame(delta_rows)

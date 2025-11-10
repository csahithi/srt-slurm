"""
Metrics calculation module for benchmark data

Key Metrics:
- Output TPS/GPU: Total output throughput divided by number of GPUs
- Output TPS/User: 1000 / Mean TPOT (ms) - represents actual per-user token generation rate
"""

import pandas as pd


def calculate_total_gpus(run: dict) -> int:
    """Calculate total number of GPUs from run configuration."""
    total_gpus = 0

    # Prefill GPUs
    if "prefill_tp" in run and "prefill_dp" in run:
        total_gpus += run["prefill_tp"] * run["prefill_dp"]

    # Decode GPUs
    if "decode_tp" in run and "decode_dp" in run:
        total_gpus += run["decode_tp"] * run["decode_dp"]

    return total_gpus if total_gpus > 0 else 1  # Default to 1 to avoid division by zero


def calculate_derived_metrics(run: dict) -> dict:
    """Calculate derived metrics for a benchmark run."""
    total_gpus = calculate_total_gpus(run)

    output_tps = run.get("output_tps", [])
    mean_tpot = run.get("mean_tpot_ms", [])

    # Calculate Output TPS/GPU
    output_tps_per_gpu = [tps / total_gpus for tps in output_tps]

    # Calculate Output TPS/User as 1000 / TPOT
    # TPOT is in milliseconds, so 1000/TPOT gives tokens/second per user
    output_tps_per_user = [1000 / tpot if tpot > 0 else 0 for tpot in mean_tpot]

    return {
        "total_gpus": total_gpus,
        "output_tps_per_gpu": output_tps_per_gpu,
        "output_tps_per_user": output_tps_per_user,
    }


def runs_to_dataframe(runs: list[dict]) -> pd.DataFrame:
    """Convert list of runs to a pandas DataFrame for easier manipulation."""
    rows = []

    for run in runs:
        metrics = calculate_derived_metrics(run)
        run_id = run.get("slurm_job_id", "Unknown")

        output_tps = run.get("output_tps", [])
        concurrencies = run.get("concurrencies", [])
        request_rates = run.get("request_rate", [])
        mean_ttft = run.get("mean_ttft_ms", [])
        mean_tpot = run.get("mean_tpot_ms", [])
        mean_itl = run.get("mean_itl_ms", [])

        # Create a row for each concurrency level
        for i in range(len(output_tps)):
            row = {
                "Run ID": run_id,
                "Run Date": run.get("run_date", "N/A"),
                "Profiler": run.get("profiler_type", "N/A"),
                "ISL": run.get("isl", "N/A"),
                "OSL": run.get("osl", "N/A"),
                "Prefill TP": run.get("prefill_tp", "N/A"),
                "Prefill DP": run.get("prefill_dp", "N/A"),
                "Decode TP": run.get("decode_tp", "N/A"),
                "Decode DP": run.get("decode_dp", "N/A"),
                "Frontends": run.get("frontends", "N/A"),
                "Total GPUs": metrics["total_gpus"],
                "Request Rate": request_rates[i] if i < len(request_rates) else "N/A",
                "Concurrency": concurrencies[i] if i < len(concurrencies) else "N/A",
                "Output TPS": output_tps[i] if i < len(output_tps) else 0,
                "Output TPS/GPU": metrics["output_tps_per_gpu"][i]
                if i < len(metrics["output_tps_per_gpu"])
                else 0,
                "Output TPS/User": metrics["output_tps_per_user"][i]
                if i < len(metrics["output_tps_per_user"])
                else 0,
                "Mean TTFT (ms)": mean_ttft[i] if i < len(mean_ttft) else "N/A",
                "Mean TPOT (ms)": mean_tpot[i] if i < len(mean_tpot) else "N/A",
                "Mean ITL (ms)": mean_itl[i] if i < len(mean_itl) else "N/A",
            }
            rows.append(row)

    return pd.DataFrame(rows)


def get_pareto_data(runs: list[dict]) -> pd.DataFrame:
    """Get data formatted for Pareto graph plotting."""
    df = runs_to_dataframe(runs)
    return df[
        [
            "Run ID",
            "Concurrency",
            "Output TPS/User",
            "Output TPS/GPU",
            "Output TPS",
            "Mean TTFT (ms)",
            "Mean TPOT (ms)",
            "Mean ITL (ms)",
            "Request Rate",
            "Total GPUs",
        ]
    ]


def get_summary_stats(runs: list[dict]) -> dict:
    """Get summary statistics for all runs."""
    df = runs_to_dataframe(runs)

    return {
        "total_runs": len(runs),
        "total_data_points": len(df),
        "unique_profilers": df["Profiler"].nunique(),
        "max_throughput": df["Output TPS"].max(),
        "max_concurrency": df["Concurrency"].max(),
    }

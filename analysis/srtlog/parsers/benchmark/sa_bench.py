# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SA-Bench benchmark output parser."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from analysis.srtlog.parsers import register_benchmark_parser

if TYPE_CHECKING:
    from analysis.srtlog.parsers import BenchmarkLaunchCommand

logger = logging.getLogger(__name__)


@register_benchmark_parser("sa-bench")
class SABenchParser:
    """Parser for SA-Bench benchmark output.
    Parses benchmark.out files and result JSON files from SA-Bench runs.
    """

    @property
    def benchmark_type(self) -> str:
        return "sa-bench"

    def parse(self, benchmark_out_path: Path) -> dict[str, Any]:
        """Parse benchmark.out file for SA-Bench results.
        Args:
            benchmark_out_path: Path to benchmark.out file
        Returns:
            Dict with aggregated benchmark results
        """
        results = {
            "benchmark_type": self.benchmark_type,
            "concurrencies": [],
            "output_tps": [],
            "mean_ttft_ms": [],
            "mean_itl_ms": [],
            "mean_tpot_ms": [],
            "p99_ttft_ms": [],
            "p99_itl_ms": [],
            "request_throughput": [],
            "completed_requests": [],
        }

        if not benchmark_out_path.exists():
            logger.warning("benchmark.out not found: %s", benchmark_out_path)
            return results

        try:
            content = benchmark_out_path.read_text()

            # Parse summary lines from benchmark output
            # Example: "Concurrency: 100, Throughput: 5000 tok/s, TTFT: 150ms, ITL: 20ms"
            concurrency_pattern = r"Concurrency[:\s]+(\d+)"
            throughput_pattern = r"(?:Output\s+)?[Tt]hroughput[:\s]+([\d.]+)"
            ttft_pattern = r"(?:Mean\s+)?TTFT[:\s]+([\d.]+)"
            itl_pattern = r"(?:Mean\s+)?ITL[:\s]+([\d.]+)"

            # Try to extract from summary lines
            for line in content.split("\n"):
                if "concurrency" in line.lower() or "throughput" in line.lower():
                    conc_match = re.search(concurrency_pattern, line, re.IGNORECASE)
                    tpt_match = re.search(throughput_pattern, line, re.IGNORECASE)
                    ttft_match = re.search(ttft_pattern, line, re.IGNORECASE)
                    itl_match = re.search(itl_pattern, line, re.IGNORECASE)

                    if conc_match and tpt_match:
                        results["concurrencies"].append(int(conc_match.group(1)))
                        results["output_tps"].append(float(tpt_match.group(1)))
                        if ttft_match:
                            results["mean_ttft_ms"].append(float(ttft_match.group(1)))
                        if itl_match:
                            results["mean_itl_ms"].append(float(itl_match.group(1)))

        except Exception as e:
            logger.warning("Failed to parse benchmark.out: %s", e)

        return results

    def parse_result_json(self, json_path: Path) -> dict[str, Any]:
        """Parse a SA-Bench result JSON file.
        Args:
            json_path: Path to result JSON (e.g., result_c100.json)
        Returns:
            Dict with benchmark metrics for this concurrency level
        """
        result = {}

        try:
            with open(json_path) as f:
                data = json.load(f)

            # Return with same field names as original JSON for compatibility
            # with downstream processing in _build_rollup_summary
            result = {
                "max_concurrency": data.get("max_concurrency"),
                # Throughput metrics (keep original field names)
                "output_throughput": data.get("output_throughput"),
                "total_token_throughput": data.get("total_token_throughput"),
                "request_throughput": data.get("request_throughput"),
                "request_goodput": data.get("request_goodput"),
                "request_rate": data.get("request_rate"),
                # Mean latencies
                "mean_ttft_ms": data.get("mean_ttft_ms"),
                "mean_tpot_ms": data.get("mean_tpot_ms"),
                "mean_itl_ms": data.get("mean_itl_ms"),
                "mean_e2el_ms": data.get("mean_e2el_ms"),
                # Median latencies
                "median_ttft_ms": data.get("median_ttft_ms"),
                "median_tpot_ms": data.get("median_tpot_ms"),
                "median_itl_ms": data.get("median_itl_ms"),
                "median_e2el_ms": data.get("median_e2el_ms"),
                # P99 latencies
                "p99_ttft_ms": data.get("p99_ttft_ms"),
                "p99_tpot_ms": data.get("p99_tpot_ms"),
                "p99_itl_ms": data.get("p99_itl_ms"),
                "p99_e2el_ms": data.get("p99_e2el_ms"),
                # Std dev latencies
                "std_ttft_ms": data.get("std_ttft_ms"),
                "std_tpot_ms": data.get("std_tpot_ms"),
                "std_itl_ms": data.get("std_itl_ms"),
                "std_e2el_ms": data.get("std_e2el_ms"),
                # Token counts
                "total_input_tokens": data.get("total_input_tokens"),
                "total_output_tokens": data.get("total_output_tokens"),
                # Metadata
                "duration": data.get("duration"),
                "completed": data.get("completed"),
                "num_prompts": data.get("num_prompts"),
            }

        except Exception as e:
            logger.warning("Failed to parse %s: %s", json_path, e)

        return result

    def parse_result_directory(self, result_dir: Path) -> list[dict[str, Any]]:
        """Parse all result JSON files in a benchmark result directory.
        Args:
            result_dir: Directory containing result_*.json files
        Returns:
            List of result dicts sorted by concurrency
        """
        results = []

        for json_file in result_dir.glob("*.json"):
            result = self.parse_result_json(json_file)
            if result.get("max_concurrency") is not None:
                results.append(result)

        # Sort by concurrency
        results.sort(key=lambda x: x.get("max_concurrency", 0) or 0)

        return results

    def find_result_directory(self, run_path: Path, isl: int | None = None, osl: int | None = None) -> Path | None:
        """Find the directory containing SA-Bench results.
        
        SA-Bench results are typically in directories named like:
        - sa-bench_isl_8192_osl_1024
        - vllm_isl_8192_osl_1024
        
        Args:
            run_path: Path to the run directory
            isl: Input sequence length
            osl: Output sequence length
            
        Returns:
            Path to results directory, or None if not found
        """
        # Search paths: run_path and run_path/logs
        search_paths = [run_path]
        logs_dir = run_path / "logs"
        if logs_dir.exists():
            search_paths.append(logs_dir)
        
        # Build prefix patterns
        if isl is not None and osl is not None:
            prefixes = [
                f"sa-bench_isl_{isl}_osl_{osl}",
                f"vllm_isl_{isl}_osl_{osl}",
            ]
        else:
            # Fallback: match any sa-bench or vllm directory
            prefixes = ["sa-bench", "vllm"]
        
        # Search for matching directories
        for search_path in search_paths:
            if not search_path.exists():
                continue
            try:
                for entry in search_path.iterdir():
                    if not entry.is_dir():
                        continue
                    # Check if directory name starts with any of our prefixes
                    for prefix in prefixes:
                        if entry.name.startswith(prefix):
                            # Verify it contains result files
                            if list(entry.glob("*.json")):
                                logger.info(f"Found SA-Bench results in: {entry}")
                                return entry
            except (OSError, PermissionError) as e:
                logger.warning(f"Error accessing {search_path}: {e}")
                continue
        
        return None


    def parse_launch_command(self, log_content: str) -> BenchmarkLaunchCommand | None:
        """Parse the SA-Bench launch command from log content.
        Looks for command lines like:
            [CMD] python -m sglang.bench_serving --model ... --base-url ...
            python -m sglang.bench_serving --model ... --base-url ...
        Also parses SA-Bench Config header format:
            SA-Bench Config: endpoint=http://localhost:8000; isl=8192; osl=1024; ...
        Args:
            log_content: Content of the benchmark log file
        Returns:
            BenchmarkLaunchCommand with parsed parameters, or None if not found
        """
        from analysis.srtlog.parsers import BenchmarkLaunchCommand

        raw_command = None

        # First, try to find [CMD] tagged command (preferred - from our scripts)
        cmd_match = re.search(r"\[CMD\]\s*(.+)$", log_content, re.MULTILINE)
        if cmd_match:
            raw_command = cmd_match.group(1).strip()

        # Fallback: pattern to match sa-bench / sglang.bench_serving command
        if not raw_command:
            command_patterns = [
                r"(python[3]?\s+-m\s+sglang\.bench_serving\s+[^\n]+)",
                r"(sa-bench\s+[^\n]+)",
                r"(python[3]?\s+.*bench_serving\.py\s+[^\n]+)",
            ]

            for pattern in command_patterns:
                match = re.search(pattern, log_content, re.IGNORECASE)
                if match:
                    raw_command = match.group(1).strip()
                    break

        # Also try SA-Bench Config header format
        if not raw_command:
            config_match = re.search(r"(SA-Bench Config:[^\n]+)", log_content)
            if config_match:
                raw_command = config_match.group(1).strip()

        if not raw_command:
            return None

        extra_args: dict[str, Any] = {}

        # Parse common arguments from command line
        arg_patterns = {
            "model": r"--model[=\s]+([^\s]+)",
            "base_url": r"--base-url[=\s]+([^\s]+)",
            "num_prompts": r"--num-prompts?[=\s]+(\d+)",
            "request_rate": r"--request-rate[=\s]+([^\s]+)",
            "max_concurrency": r"--max-concurrency[=\s]+(\d+)",
            "input_len": r"--(?:input-len|random-input-len)[=\s]+(\d+)",
            "output_len": r"--(?:output-len|random-output-len)[=\s]+(\d+)",
            "dataset": r"--dataset[=\s]+([^\s]+)",
            "dataset_path": r"--dataset-path[=\s]+([^\s]+)",
        }

        for field, pattern in arg_patterns.items():
            match = re.search(pattern, raw_command)
            if match:
                value: Any = match.group(1)
                # Convert to appropriate type
                if field in ("num_prompts", "max_concurrency", "input_len", "output_len"):
                    value = int(value)
                elif field == "request_rate" and value != "inf":
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                extra_args[field] = value

        # Also parse from SA-Bench Config header format
        # Format: SA-Bench Config: endpoint=http://localhost:8000; isl=8192; osl=1024; concurrencies=28; req_rate=inf; model=dsr1-fp8
        header_patterns = {
            "base_url": r"endpoint=([^;\s]+)",
            "model": r"model=([^;\s]+)",
            "input_len": r"isl=(\d+)",
            "output_len": r"osl=(\d+)",
            "max_concurrency": r"concurrencies=(\d+)",
            "request_rate": r"req_rate=([^;\s]+)",
        }

        for field, pattern in header_patterns.items():
            if field not in extra_args:
                match = re.search(pattern, raw_command)
                if match:
                    value = match.group(1)
                    if field in ("input_len", "output_len", "max_concurrency"):
                        value = int(value)
                    elif field == "request_rate" and value != "inf":
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                    extra_args[field] = value

        return BenchmarkLaunchCommand(
            benchmark_type=self.benchmark_type,
            raw_command=raw_command,
            extra_args=extra_args,
        )

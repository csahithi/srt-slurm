# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mooncake Router benchmark output parser."""

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


@register_benchmark_parser("mooncake-router")
class MooncakeRouterParser:
    """Parser for Mooncake Router benchmark output.

    Parses benchmark.out files and AIPerf result JSON files from mooncake-router runs.
    """

    @property
    def benchmark_type(self) -> str:
        return "mooncake-router"

    def parse(self, benchmark_out_path: Path) -> dict[str, Any]:
        """Parse benchmark.out file for mooncake-router results.

        Args:
            benchmark_out_path: Path to benchmark.out file

        Returns:
            Dict with aggregated benchmark results
        """
        results = {
            "benchmark_type": self.benchmark_type,
            "output_tps": None,
            "request_throughput": None,
            "mean_ttft_ms": None,
            "mean_itl_ms": None,
            "total_requests": None,
        }

        if not benchmark_out_path.exists():
            logger.warning("benchmark.out not found: %s", benchmark_out_path)
            return results

        try:
            content = benchmark_out_path.read_text()

            # Parse mooncake-router output patterns
            # Example: "Request throughput: 3.37 req/s"
            # Example: "Output token throughput: 1150.92 tok/s"
            req_tpt_pattern = r"[Rr]equest\s+throughput[:\s]+([\d.]+)"
            out_tpt_pattern = r"[Oo]utput\s+(?:token\s+)?throughput[:\s]+([\d.]+)"
            ttft_pattern = r"[Tt]ime\s+to\s+first\s+token[:\s]+([\d.]+)"
            itl_pattern = r"[Ii]nter.?token\s+latency[:\s]+([\d.]+)"

            for line in content.split("\n"):
                if req_tpt_match := re.search(req_tpt_pattern, line):
                    results["request_throughput"] = float(req_tpt_match.group(1))
                if out_tpt_match := re.search(out_tpt_pattern, line):
                    results["output_tps"] = float(out_tpt_match.group(1))
                if ttft_match := re.search(ttft_pattern, line):
                    results["mean_ttft_ms"] = float(ttft_match.group(1))
                if itl_match := re.search(itl_pattern, line):
                    results["mean_itl_ms"] = float(itl_match.group(1))

        except Exception as e:
            logger.warning("Failed to parse benchmark.out: %s", e)

        return results

    def parse_result_json(self, json_path: Path) -> dict[str, Any]:
        """Parse an AIPerf result JSON file.

        Args:
            json_path: Path to profile_export_aiperf.json

        Returns:
            Dict with benchmark metrics
        """
        result = {}

        try:
            with open(json_path) as f:
                data = json.load(f)

            # AIPerf format has nested structure with unit and values
            result = {
                "concurrency": 0,  # Mooncake uses open-loop, no fixed concurrency
                # Throughput metrics
                "output_tps": self._get_metric(data, "output_token_throughput", "avg"),
                "request_throughput": self._get_metric(data, "request_throughput", "avg"),
                # Mean latencies (convert from ms)
                "mean_ttft_ms": self._get_metric(data, "time_to_first_token", "avg"),
                "mean_tpot_ms": self._get_metric(data, "inter_token_latency", "avg"),
                "mean_itl_ms": self._get_metric(data, "inter_token_latency", "avg"),
                "mean_e2el_ms": self._get_metric(data, "request_latency", "avg"),
                # Median latencies
                "median_ttft_ms": self._get_metric(data, "time_to_first_token", "p50"),
                "median_tpot_ms": self._get_metric(data, "inter_token_latency", "p50"),
                "median_itl_ms": self._get_metric(data, "inter_token_latency", "p50"),
                "median_e2el_ms": self._get_metric(data, "request_latency", "p50"),
                # P99 latencies
                "p99_ttft_ms": self._get_metric(data, "time_to_first_token", "p99"),
                "p99_tpot_ms": self._get_metric(data, "inter_token_latency", "p99"),
                "p99_itl_ms": self._get_metric(data, "inter_token_latency", "p99"),
                "p99_e2el_ms": self._get_metric(data, "request_latency", "p99"),
                # Std dev latencies
                "std_ttft_ms": self._get_metric(data, "time_to_first_token", "std"),
                "std_itl_ms": self._get_metric(data, "inter_token_latency", "std"),
                "std_e2el_ms": self._get_metric(data, "request_latency", "std"),
                # Request count
                "completed": self._get_metric(data, "request_count", "avg"),
                "num_prompts": self._get_metric(data, "request_count", "avg"),
            }

            # Also extract per-user throughput if available
            tps_per_user = self._get_metric(data, "output_token_throughput_per_user", "avg")
            if tps_per_user:
                result["output_tps_per_user"] = tps_per_user

        except Exception as e:
            logger.warning("Failed to parse %s: %s", json_path, e)

        return result

    def _get_metric(self, data: dict, metric_name: str, stat: str) -> float | None:
        """Extract a metric value from AIPerf data structure.

        Args:
            data: AIPerf JSON data
            metric_name: Name of the metric (e.g., "time_to_first_token")
            stat: Statistic to extract (e.g., "avg", "p50", "p99")

        Returns:
            Metric value or None if not found
        """
        try:
            metric_data = data.get(metric_name, {})
            if isinstance(metric_data, dict):
                value = metric_data.get(stat)
                if value is not None:
                    return float(value)
        except (KeyError, TypeError, ValueError):
            pass
        return None

    def parse_result_directory(self, result_dir: Path) -> list[dict[str, Any]]:
        """Parse AIPerf result files in a directory.

        Args:
            result_dir: Directory containing profile_export_aiperf.json

        Returns:
            List of result dicts (usually just one for mooncake-router)
        """
        results = []

        # Look for AIPerf JSON files
        for json_file in result_dir.rglob("profile_export_aiperf.json"):
            result = self.parse_result_json(json_file)
            if result.get("output_tps") is not None:
                results.append(result)

        return results

    def find_aiperf_results(self, log_dir: Path) -> list[Path]:
        """Find all AIPerf result files in a log directory.

        Args:
            log_dir: Root log directory

        Returns:
            List of paths to profile_export_aiperf.json files
        """
        return list(log_dir.rglob("profile_export_aiperf.json"))

    def parse_launch_command(self, log_content: str) -> BenchmarkLaunchCommand | None:
        """Parse the mooncake-router launch command from log content.

        Looks for command lines like:
            [CMD] aiperf profile --model ... --url ...
            genai-perf profile --model ... --endpoint ...

        Also parses header format:
            Endpoint: http://localhost:8000
            Model: Qwen/Qwen3-32B
            Workload: conversation

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

        # Fallback: pattern to match genai-perf, aiperf or mooncake-router commands
        # aiperf format: aiperf profile -m "Model" --url "http://..." --concurrency 10
        if not raw_command:
            command_patterns = [
                r"(aiperf\s+profile\s+[^\n]+)",
                r"(genai-perf\s+profile\s+[^\n]+)",
                r"(python[3]?\s+.*genai_perf[^\n]+)",
                r"(python[3]?\s+.*aiperf[^\n]+)",
                r"(mooncake-router\s+[^\n]+)",
            ]

            for pattern in command_patterns:
                match = re.search(pattern, log_content, re.IGNORECASE)
                if match:
                    raw_command = match.group(1).strip()
                    break

        # If no command found, try to build from header format
        if not raw_command:
            if "Mooncake Router Benchmark" in log_content:
                raw_command = "mooncake-router-benchmark (from header)"

        if not raw_command:
            return None

        extra_args: dict[str, Any] = {}

        # Parse aiperf/genai-perf arguments from command line
        # Supports both --model and -m formats, quoted and unquoted values
        arg_patterns = {
            "model": r"(?:--model|-m)[=\s]+[\"']?([^\"'\s]+)[\"']?",
            "base_url": r"--url[=\s]+[\"']?([^\"'\s]+)[\"']?",
            "num_prompts": r"--(?:num-prompts|request-count|request)[=\s]+(\d+)",
            "request_rate": r"--request-rate[=\s]+([^\s]+)",
            "max_concurrency": r"--concurrency[=\s]+(\d+)",
            "input_len": r"--(?:synthetic-input-tokens-mean|input-sequence-length|isl)[=\s]+(\d+)",
            "output_len": r"--(?:output-tokens-mean|output-sequence-length|osl)[=\s]+(\d+)",
        }

        for field, pattern in arg_patterns.items():
            match = re.search(pattern, raw_command)
            if match:
                value: Any = match.group(1)
                if field in ("num_prompts", "max_concurrency", "input_len", "output_len"):
                    value = int(value)
                elif field == "request_rate" and value != "inf":
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                extra_args[field] = value

        # Also parse from header format (srtctl-style)
        header_patterns = {
            "model": r"^Model:\s*(.+)$",
            "base_url": r"^Endpoint:\s*(.+)$",
            "dataset": r"^Workload:\s*(.+)$",
        }

        for field, pattern in header_patterns.items():
            if field not in extra_args:
                match = re.search(pattern, log_content, re.MULTILINE)
                if match:
                    extra_args[field] = match.group(1).strip()

        return BenchmarkLaunchCommand(
            benchmark_type=self.benchmark_type,
            raw_command=raw_command,
            extra_args=extra_args,
        )


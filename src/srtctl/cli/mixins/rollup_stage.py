# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Rollup stage mixin for SweepOrchestrator.

Aggregates experiment data from multiple benchmark runs into a single consolidated summary.
Includes node-level metrics parsed from prefill/decode .out and .err files using analysis.srtlog.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from srtctl.cli.mixins.rollup import (
    EnvironmentConfig,
    LaunchCommandRollup,
    NodesSummary,
    RollupResult,
    RollupSummary,
)

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig
    from srtctl.core.topology import Endpoint

logger = logging.getLogger(__name__)


class RollupStageMixin:
    """Mixin for rollup stage that consolidates experiment data.

    Requires:
        self.config: SrtConfig
        self.runtime: RuntimeContext
        self.endpoints: list[Endpoint]
    """

    # Type hints for mixin dependencies
    config: SrtConfig
    runtime: RuntimeContext

    @property
    def endpoints(self) -> list[Endpoint]:
        """Endpoint allocation topology."""
        ...

    def run_rollup(self, tags: list[str] | None = None) -> Path | None:
        """Run the rollup stage to consolidate experiment data.

        Args:
            tags: Optional list of tags for the experiment

        Returns:
            Path to the generated rollup.json file, or None if rollup failed
        """
        logger.info("Running rollup stage")

        try:
            # Collect benchmark results
            results = self._collect_benchmark_results()

            if not results:
                logger.warning("No benchmark results found to rollup")
                return None

            # Collect node metrics using analysis.srtlog
            nodes_summary = self._collect_node_metrics()

            # Collect benchmark launch command
            benchmark_command = self._collect_benchmark_command()

            # Collect environment and engine configuration
            environment_config = self._collect_environment_config()

            # Build rollup summary
            summary = self._build_rollup_summary(results, tags, nodes_summary, benchmark_command, environment_config)

            # Write rollup.json
            rollup_path = self.runtime.log_dir / "rollup.json"
            self._write_rollup(summary, rollup_path)

            logger.info("Rollup complete: %s", rollup_path)
            logger.info(
                "Summary: %d results, max output TPS: %.2f, %d nodes",
                len(summary.results),
                summary.max_output_tps or 0,
                len(nodes_summary.nodes) if nodes_summary else 0,
            )

            return rollup_path

        except Exception as e:
            logger.error("Rollup failed: %s", e)
            return None

    def _collect_benchmark_results(self) -> list[dict[str, Any]]:
        """Collect all benchmark result JSON files from the log directory.

        Uses the appropriate benchmark parser based on config.benchmark.type.

        Returns:
            List of parsed benchmark result dicts
        """
        results = []
        benchmark_type = self.config.benchmark.type

        try:
            from analysis.srtlog.parsers import get_benchmark_parser, list_benchmark_parsers

            # Get the appropriate parser
            try:
                parser = get_benchmark_parser(benchmark_type)
                logger.debug("Using %s benchmark parser", benchmark_type)
            except ValueError:
                logger.warning(
                    "No parser for benchmark type '%s', available: %s. Using fallback.",
                    benchmark_type,
                    list_benchmark_parsers(),
                )
                parser = None

            # Try parser-specific result collection first
            if parser is not None:
                # For mooncake-router, look for AIPerf results
                if hasattr(parser, "find_aiperf_results"):
                    aiperf_files = parser.find_aiperf_results(self.runtime.log_dir)
                    for aiperf_path in aiperf_files:
                        result = parser.parse_result_json(aiperf_path)
                        if result.get("output_tps") is not None:
                            results.append(result)
                            logger.debug("Loaded AIPerf result: %s", aiperf_path)

                # For sa-bench style, look for result directories
                if hasattr(parser, "parse_result_directory"):
                    for entry in self.runtime.log_dir.iterdir():
                        if not entry.is_dir():
                            continue
                        # Match patterns like sa-bench_isl_X_osl_Y
                        if "_isl_" in entry.name and "_osl_" in entry.name:
                            logger.debug("Found benchmark results directory: %s", entry.name)
                            dir_results = parser.parse_result_directory(entry)
                            results.extend(dir_results)

        except ImportError:
            logger.debug("analysis.srtlog.parsers not available, using fallback")
            parser = None

        # Fallback: direct JSON parsing
        if not results:
            for entry in self.runtime.log_dir.iterdir():
                if not entry.is_dir():
                    continue

                # Match patterns like sa-bench_isl_X_osl_Y, vllm_isl_X_osl_Y
                if "_isl_" in entry.name and "_osl_" in entry.name:
                    logger.debug("Found benchmark results directory: %s", entry.name)

                    # Parse all JSON files in the directory
                    for json_file in entry.glob("*.json"):
                        try:
                            with open(json_file) as f:
                                data = json.load(f)
                                results.append(data)
                                logger.debug("Loaded result: %s", json_file.name)
                        except Exception as e:
                            logger.warning("Failed to parse %s: %s", json_file, e)

        # Sort by concurrency
        results.sort(key=lambda x: x.get("max_concurrency", 0) or 0)

        logger.info("Collected %d benchmark results", len(results))
        return results

    def _collect_node_metrics(self) -> NodesSummary | None:
        """Collect node metrics from prefill/decode log files.

        Uses the appropriate node parser based on config.backend_type.
        Falls back through parser versions if needed (e.g., sglang -> sglang-v2).

        Returns:
            NodesSummary with aggregated node statistics, or None if parsing fails
        """
        backend_type = self.config.backend_type
        log_dir = self.runtime.log_dir

        try:
            from analysis.srtlog.parsers import get_node_parser

            # Try parsers in order of preference
            parser_order = self._get_parser_order(backend_type)
            logger.debug("Parser order for %s: %s", backend_type, parser_order)

            nodes = []
            used_parser = None
            parser = None

            for parser_type in parser_order:
                try:
                    parser = get_node_parser(parser_type)
                    nodes = parser.parse_logs(log_dir)

                    # Check if we got meaningful results (batches or config)
                    total_batches = sum(len(n.batches) for n in nodes)
                    has_config = any(n.config for n in nodes)
                    if total_batches > 0 or has_config:
                        used_parser = parser_type
                        logger.info("Using %s parser: found %d nodes with %d batches", parser_type, len(nodes), total_batches)
                        break
                    else:
                        logger.debug("%s parser found no batches, trying next", parser_type)

                except ValueError:
                    logger.debug("Parser %s not available", parser_type)
                    continue

            if not nodes:
                logger.warning("No node metrics found in %s with any parser", log_dir)
                return None

            # Build summary from parsed nodes
            summary = NodesSummary.from_node_metrics_list(nodes)

            # Parse launch commands for each node
            if parser is not None and hasattr(parser, "parse_launch_command"):
                self._add_launch_commands_to_summary(summary, parser, log_dir)

            if summary.total_agg_nodes > 0:
                logger.info("Node summary (%s): %d agg nodes", used_parser, summary.total_agg_nodes)
            else:
                logger.info(
                    "Node summary (%s): %d prefill, %d decode nodes",
                    used_parser,
                    summary.total_prefill_nodes,
                    summary.total_decode_nodes,
                )

            return summary

        except ImportError:
            logger.warning("analysis.srtlog.parsers not available, skipping node metrics")
            return None
        except Exception as e:
            logger.warning("Failed to collect node metrics: %s", e)
            return None

    def _add_launch_commands_to_summary(self, summary: NodesSummary, parser: Any, log_dir: Path) -> None:
        """Parse and add launch commands to each node in the summary.

        Args:
            summary: NodesSummary to update
            parser: Node parser with parse_launch_command method
            log_dir: Directory containing log files
        """
        for node_rollup in summary.nodes:
            # Find the log file for this node
            node_name = node_rollup.node_name
            worker_type = node_rollup.worker_type
            worker_id = node_rollup.worker_id

            # Try both .out and .err files
            for ext in [".out", ".err"]:
                log_file = log_dir / f"{node_name}_{worker_type}_{worker_id}{ext}"
                if log_file.exists():
                    try:
                        content = log_file.read_text(errors="replace")
                        cmd = parser.parse_launch_command(content, worker_type=worker_type)
                        if cmd:
                            args = cmd.extra_args
                            node_rollup.launch_command = LaunchCommandRollup(
                                raw_command=cmd.raw_command,
                                command_type="worker",
                                model_path=args.get("model_path"),
                                served_model_name=args.get("served_model_name"),
                                worker_type=worker_type,
                                backend_type=cmd.backend_type,
                                disaggregation_mode=args.get("disaggregation_mode"),
                                tp_size=args.get("tp_size"),
                                pp_size=args.get("pp_size"),
                                dp_size=args.get("dp_size"),
                                ep_size=args.get("ep_size"),
                                port=args.get("port"),
                                max_num_seqs=args.get("max_num_seqs"),
                                max_model_len=args.get("max_model_len"),
                            )
                            logger.debug("Parsed launch command for %s_%s_%s", node_name, worker_type, worker_id)
                            break
                    except Exception as e:
                        logger.debug("Failed to parse launch command from %s: %s", log_file, e)

    def _collect_benchmark_command(self) -> LaunchCommandRollup | None:
        """Parse the benchmark launch command from benchmark.out.

        Returns:
            LaunchCommandRollup with benchmark parameters, or None if not found
        """
        benchmark_type = self.config.benchmark.type
        log_dir = self.runtime.log_dir

        try:
            from analysis.srtlog.parsers import get_benchmark_parser

            parser = get_benchmark_parser(benchmark_type)

            # Look for benchmark.out file
            benchmark_out = log_dir / "benchmark.out"
            if not benchmark_out.exists():
                logger.debug("benchmark.out not found in %s", log_dir)
                return None

            content = benchmark_out.read_text(errors="replace")
            cmd = parser.parse_launch_command(content)

            if cmd:
                args = cmd.extra_args
                return LaunchCommandRollup(
                    raw_command=cmd.raw_command,
                    command_type="benchmark",
                    model_path=args.get("model"),
                    benchmark_type=cmd.benchmark_type,
                    base_url=args.get("base_url"),
                    max_concurrency=args.get("max_concurrency"),
                    num_prompts=args.get("num_prompts"),
                    input_len=args.get("input_len"),
                    output_len=args.get("output_len"),
                )

        except ImportError:
            logger.debug("analysis.srtlog.parsers not available")
        except ValueError as e:
            logger.debug("No benchmark parser for %s: %s", benchmark_type, e)
        except Exception as e:
            logger.debug("Failed to parse benchmark command: %s", e)

        return None

    def _collect_environment_config(self) -> EnvironmentConfig | None:
        """Collect environment variables and engine config from config files.

        Parses:
        1. config.yaml for prefill_environment and decode_environment
        2. YAML config files (e.g., trtllm_config_prefill.yaml) for engine settings

        Returns:
            EnvironmentConfig with environment variables and engine config, or None if not found
        """
        log_dir = self.runtime.log_dir

        try:
            import yaml
        except ImportError:
            logger.debug("PyYAML not available, skipping environment config collection")
            return None

        config = EnvironmentConfig()

        # Try to find config.yaml in the job output directory
        # It could be in log_dir, log_dir.parent, or a sibling directory
        config_paths = [
            log_dir / "config.yaml",
            log_dir.parent / "config.yaml",
            log_dir.parent.parent / "config.yaml",
        ]

        config_yaml = None
        for path in config_paths:
            if path.exists():
                config_yaml = path
                break

        if config_yaml:
            try:
                with open(config_yaml) as f:
                    job_config = yaml.safe_load(f)

                backend_section = job_config.get("backend", {})

                # Extract environment variables
                if "prefill_environment" in backend_section:
                    config.prefill_environment = backend_section["prefill_environment"]
                    logger.debug("Found prefill_environment with %d vars", len(config.prefill_environment))

                if "decode_environment" in backend_section:
                    config.decode_environment = backend_section["decode_environment"]
                    logger.debug("Found decode_environment with %d vars", len(config.decode_environment))

                if "aggregated_environment" in backend_section:
                    config.aggregated_environment = backend_section["aggregated_environment"]
                    logger.debug("Found aggregated_environment with %d vars", len(config.aggregated_environment))

                # For TRTLLM, also extract inline engine config
                if "trtllm_config" in backend_section:
                    trtllm_config = backend_section["trtllm_config"]
                    if "prefill" in trtllm_config:
                        config.prefill_engine_config = trtllm_config["prefill"]
                    if "decode" in trtllm_config:
                        config.decode_engine_config = trtllm_config["decode"]
                    if "aggregated" in trtllm_config:
                        config.aggregated_engine_config = trtllm_config["aggregated"]

                # For SGLang, extract sglang_config if present
                if "sglang_config" in backend_section:
                    sglang_config = backend_section["sglang_config"]
                    if "prefill" in sglang_config:
                        config.prefill_engine_config = sglang_config["prefill"]
                    if "decode" in sglang_config:
                        config.decode_engine_config = sglang_config["decode"]
                    if "aggregated" in sglang_config:
                        config.aggregated_engine_config = sglang_config["aggregated"]

            except Exception as e:
                logger.debug("Failed to parse config.yaml: %s", e)

        # Also look for separate YAML config files (e.g., trtllm_config_prefill.yaml)
        prefill_yaml = log_dir / "trtllm_config_prefill.yaml"
        decode_yaml = log_dir / "trtllm_config_decode.yaml"

        if prefill_yaml.exists() and not config.prefill_engine_config:
            try:
                with open(prefill_yaml) as f:
                    config.prefill_engine_config = yaml.safe_load(f)
                logger.debug("Loaded prefill engine config from %s", prefill_yaml)
            except Exception as e:
                logger.debug("Failed to parse %s: %s", prefill_yaml, e)

        if decode_yaml.exists() and not config.decode_engine_config:
            try:
                with open(decode_yaml) as f:
                    config.decode_engine_config = yaml.safe_load(f)
                logger.debug("Loaded decode engine config from %s", decode_yaml)
            except Exception as e:
                logger.debug("Failed to parse %s: %s", decode_yaml, e)

        # Return None if we didn't find anything
        if not any([
            config.prefill_environment,
            config.decode_environment,
            config.aggregated_environment,
            config.prefill_engine_config,
            config.decode_engine_config,
            config.aggregated_engine_config,
        ]):
            logger.debug("No environment or engine config found")
            return None

        # Log what we found
        env_counts = []
        if config.prefill_environment:
            env_counts.append(f"{len(config.prefill_environment)} prefill")
        if config.decode_environment:
            env_counts.append(f"{len(config.decode_environment)} decode")
        if config.aggregated_environment:
            env_counts.append(f"{len(config.aggregated_environment)} agg")

        if env_counts:
            logger.info("Collected environment vars: %s", ", ".join(env_counts))

        return config

    def _get_parser_order(self, backend_type: str) -> list[str]:
        """Get the order of parsers to try for a given backend type.

        Args:
            backend_type: Backend type from config (e.g., "sglang", "trtllm")

        Returns:
            List of parser types to try in order
        """
        parser_orders = {
            "sglang": ["sglang"],
            "trtllm": ["trtllm"],
        }

        return parser_orders.get(backend_type, [backend_type])

    def _build_rollup_summary(
        self,
        results: list[dict[str, Any]],
        tags: list[str] | None = None,
        nodes_summary: NodesSummary | None = None,
        benchmark_command: LaunchCommandRollup | None = None,
        environment_config: EnvironmentConfig | None = None,
    ) -> RollupSummary:
        """Build a RollupSummary from collected results.

        Args:
            results: List of parsed benchmark result dicts
            tags: Optional tags for the experiment
            nodes_summary: Optional node-level metrics summary
            benchmark_command: Optional parsed benchmark launch command
            environment_config: Optional environment and engine configuration

        Returns:
            RollupSummary instance
        """
        r = self.config.resources
        b = self.config.benchmark

        # Determine topology
        is_disaggregated = r.is_disaggregated

        if is_disaggregated:
            total_gpus = r.prefill_gpus + r.decode_gpus
        else:
            total_gpus = (r.agg_nodes or 1) * r.gpus_per_node

        # Build summary
        summary = RollupSummary(
            # Identification
            job_id=self.runtime.job_id,
            job_name=self.config.name,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # Model config
            model_path=str(self.runtime.model_path),
            model_name=self.config.served_model_name,
            precision=self.config.model.precision,
            gpu_type=r.gpu_type,
            gpus_per_node=r.gpus_per_node,
            backend_type=self.config.backend_type,
            frontend_type=self.config.frontend.type,
            # Resource allocation
            is_disaggregated=is_disaggregated,
            total_nodes=r.total_nodes,
            total_gpus=total_gpus,
            # Benchmark config
            benchmark_type=b.type,
            isl=b.isl,
            osl=b.osl,
            concurrencies=b.get_concurrency_list(),
            # Node metrics
            nodes_summary=nodes_summary,
            # Environment and engine configuration
            environment_config=environment_config,
            # Launch commands
            benchmark_command=benchmark_command,
            # Tags
            tags=tags or [],
        )

        # Add disaggregated-specific fields
        if is_disaggregated:
            summary.prefill_nodes = r.prefill_nodes
            summary.decode_nodes = r.decode_nodes
            summary.prefill_workers = r.num_prefill
            summary.decode_workers = r.num_decode
            summary.prefill_gpus = r.prefill_gpus
            summary.decode_gpus = r.decode_gpus
        else:
            summary.agg_nodes = r.agg_nodes
            summary.agg_workers = r.num_agg

        # Convert results to RollupResult objects
        for data in results:
            result = RollupResult(
                concurrency=data.get("max_concurrency", 0),
                output_tps=data.get("output_throughput", 0),
                total_tps=data.get("total_token_throughput"),
                request_throughput=data.get("request_throughput"),
                request_goodput=data.get("request_goodput"),
                request_rate=data.get("request_rate"),
                # Mean latencies
                mean_ttft_ms=data.get("mean_ttft_ms"),
                mean_tpot_ms=data.get("mean_tpot_ms"),
                mean_itl_ms=data.get("mean_itl_ms"),
                mean_e2el_ms=data.get("mean_e2el_ms"),
                # Median latencies
                median_ttft_ms=data.get("median_ttft_ms"),
                median_tpot_ms=data.get("median_tpot_ms"),
                median_itl_ms=data.get("median_itl_ms"),
                median_e2el_ms=data.get("median_e2el_ms"),
                # P99 latencies
                p99_ttft_ms=data.get("p99_ttft_ms"),
                p99_tpot_ms=data.get("p99_tpot_ms"),
                p99_itl_ms=data.get("p99_itl_ms"),
                p99_e2el_ms=data.get("p99_e2el_ms"),
                # Token counts
                total_input_tokens=data.get("total_input_tokens"),
                total_output_tokens=data.get("total_output_tokens"),
                # Metadata
                duration=data.get("duration"),
                completed=data.get("completed"),
                num_prompts=data.get("num_prompts"),
            )
            summary.results.append(result)

        # Compute summary statistics
        summary.compute_summary_stats()

        return summary

    def _write_rollup(self, summary: RollupSummary, path: Path) -> None:
        """Write rollup summary to JSON file.

        Args:
            summary: RollupSummary to write
            path: Output file path
        """
        # Convert to dict, handling nested dataclasses
        data = asdict(summary)

        # Write with nice formatting
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.debug("Wrote rollup to %s", path)


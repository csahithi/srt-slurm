# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for RunLoader integration with parsers.

Tests that the RunLoader correctly uses the parser infrastructure.
"""

import json
import tempfile
from pathlib import Path

import pytest

from analysis.srtlog.run_loader import RunLoader
from tests.fixtures_parsers import ParserTestHarness, SampleSABenchData


class TestRunLoaderWithParsers:
    """Test RunLoader integration with parser infrastructure."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_run_metadata(self):
        """Sample run metadata JSON."""
        return {
            "job_id": "12345",
            "job_name": "test_run",
            "generated_at": "20250126_120000",
            "model": {
                "path": "/models/test",
                "container": "sglang:latest",
            },
            "resources": {
                "prefill_nodes": 1,
                "decode_nodes": 1,
                "prefill_workers": 2,
                "decode_workers": 4,
                "agg_workers": 0,
                "gpus_per_node": 8,
                "gpu_type": "H100",
            },
            "benchmark": {
                "type": "sa-bench",
                "isl": "8192",
                "osl": "1024",
                "concurrencies": "50x100x200",
                "req-rate": "inf",
            },
            "tags": ["test"],
        }

    def test_parse_sa_bench_with_parser(self, temp_dir, sample_run_metadata):
        """Test that RunLoader uses SA-Bench parser correctly."""
        # Create run directory
        run_dir = temp_dir / "12345_2P_4D_20250126_120000"
        run_dir.mkdir()

        # Create metadata JSON
        metadata_path = run_dir / "12345.json"
        with open(metadata_path, "w") as f:
            json.dump(sample_run_metadata, f)

        # Create benchmark results using test harness
        bench_dir = run_dir / "sa-bench_isl_8192_osl_1024"
        bench_dir.mkdir()

        # Create result JSON files directly in bench_dir
        for concurrency in [50, 100, 200]:
            result_data = SampleSABenchData.result_json(concurrency)
            result_path = bench_dir / f"result_c{concurrency}.json"
            with open(result_path, "w") as f:
                json.dump(result_data, f)

        # Load the run
        loader = RunLoader(str(temp_dir))
        run = loader.load_single("12345_2P_4D_20250126_120000")

        # Verify run was loaded
        assert run is not None
        assert run.job_id == "12345"

        # Verify benchmark results were parsed
        assert len(run.profiler.output_tps) == 3
        assert run.profiler.output_tps[0] == 2500.0  # 50 * 50
        assert run.profiler.output_tps[1] == 5000.0  # 100 * 50
        assert run.profiler.output_tps[2] == 10000.0  # 200 * 50

        # Verify concurrencies
        assert run.profiler.concurrency_values == [50, 100, 200]

    def test_load_all_runs_with_parsers(self, temp_dir, sample_run_metadata):
        """Test loading multiple runs with parser infrastructure."""
        # Create multiple run directories
        for job_id in [12345, 12346]:
            run_dir = temp_dir / f"{job_id}_2P_4D_20250126_120000"
            run_dir.mkdir()

            # Create metadata
            metadata = sample_run_metadata.copy()
            metadata["job_id"] = str(job_id)
            metadata_path = run_dir / f"{job_id}.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

            # Create benchmark results
            bench_dir = run_dir / "sa-bench_isl_8192_osl_1024"
            bench_dir.mkdir()

            for concurrency in [50, 100]:
                result_data = SampleSABenchData.result_json(concurrency)
                result_path = bench_dir / f"result_c{concurrency}.json"
                with open(result_path, "w") as f:
                    json.dump(result_data, f)

        # Load all runs
        loader = RunLoader(str(temp_dir))
        runs = loader.load_all()

        # Verify both runs were loaded
        assert len(runs) == 2
        job_ids = {run.job_id for run in runs}
        assert "12345" in job_ids
        assert "12346" in job_ids

        # Verify each run has benchmark data
        for run in runs:
            assert len(run.profiler.output_tps) == 2

    def test_parser_fallback_to_manual(self, temp_dir, sample_run_metadata):
        """Test fallback to manual parsing when parser fails."""
        # Create run directory
        run_dir = temp_dir / "12345_2P_4D_20250126_120000"
        run_dir.mkdir()

        # Create metadata
        metadata_path = run_dir / "12345.json"
        with open(metadata_path, "w") as f:
            json.dump(sample_run_metadata, f)

        # Create benchmark results with unknown benchmark type
        bench_dir = run_dir / "unknown-bench_isl_8192_osl_1024"
        bench_dir.mkdir()

        # Create result JSON file
        result_data = SampleSABenchData.result_json(100)
        result_path = bench_dir / "result_c100.json"
        with open(result_path, "w") as f:
            json.dump(result_data, f)

        # Load the run - should fall back to manual parsing
        loader = RunLoader(str(temp_dir))
        run = loader.load_single("12345_2P_4D_20250126_120000")

        # Verify run was loaded with manual parser
        assert run is not None
        # Note: fallback won't find results in unknown-bench directory
        # but it shouldn't crash

    def test_load_node_metrics_sglang(self, temp_dir, sample_run_metadata):
        """Test loading node metrics for SGLang runs."""
        # Create run directory
        run_dir = temp_dir / "12345_2P_4D_20250126_120000"
        run_dir.mkdir()

        # Create metadata
        metadata = sample_run_metadata.copy()
        metadata["model"]["container"] = "sglang:latest"
        metadata_path = run_dir / "12345.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # Create logs subdirectory
        logs_dir = run_dir / "logs"
        logs_dir.mkdir()

        # Create SGLang node logs using test harness
        ParserTestHarness.create_sglang_node_logs(logs_dir, num_prefill=2, num_decode=4)

        # Load node metrics
        loader = RunLoader(str(temp_dir))
        nodes = loader.load_node_metrics(str(run_dir), backend_type="sglang")

        # Verify nodes were loaded
        assert len(nodes) == 6  # 2 prefill + 4 decode
        worker_types = [node.worker_type for node in nodes]
        assert worker_types.count("prefill") == 2
        assert worker_types.count("decode") == 4

    def test_load_node_metrics_trtllm(self, temp_dir, sample_run_metadata):
        """Test loading node metrics for TRTLLM runs."""
        # Create run directory
        run_dir = temp_dir / "12345_2P_4D_20250126_120000"
        run_dir.mkdir()

        # Create metadata
        metadata = sample_run_metadata.copy()
        metadata["model"]["container"] = "trtllm:latest"
        metadata_path = run_dir / "12345.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # Create logs subdirectory
        logs_dir = run_dir / "logs"
        logs_dir.mkdir()

        # Create TRTLLM node logs using test harness
        ParserTestHarness.create_trtllm_node_logs(logs_dir, num_prefill=2, num_decode=4)

        # Load node metrics
        loader = RunLoader(str(temp_dir))
        nodes = loader.load_node_metrics(str(run_dir), backend_type="trtllm")

        # Verify nodes were loaded
        assert len(nodes) == 6  # 2 prefill + 4 decode
        worker_types = [node.worker_type for node in nodes]
        assert worker_types.count("prefill") == 2
        assert worker_types.count("decode") == 4

    def test_load_node_metrics_for_run(self, temp_dir, sample_run_metadata):
        """Test loading node metrics with automatic backend detection."""
        # Create run directory
        run_dir = temp_dir / "12345_2P_4D_20250126_120000"
        run_dir.mkdir()

        # Create metadata with SGLang container
        metadata = sample_run_metadata.copy()
        metadata["model"]["container"] = "sglang:v0.2.0"
        metadata_path = run_dir / "12345.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # Create benchmark results
        bench_dir = run_dir / "sa-bench_isl_8192_osl_1024"
        bench_dir.mkdir()
        result_data = SampleSABenchData.result_json(100)
        result_path = bench_dir / "result_c100.json"
        with open(result_path, "w") as f:
            json.dump(result_data, f)

        # Create logs subdirectory with SGLang logs
        logs_dir = run_dir / "logs"
        logs_dir.mkdir()
        ParserTestHarness.create_sglang_node_logs(logs_dir, num_prefill=1, num_decode=2)

        # Load the run
        loader = RunLoader(str(temp_dir))
        run = loader.load_single("12345_2P_4D_20250126_120000")

        # Load node metrics with automatic detection
        nodes = loader.load_node_metrics_for_run(run)

        # Verify nodes were loaded
        assert len(nodes) == 3  # 1 prefill + 2 decode

    def test_convert_parser_results_to_dict(self, temp_dir):
        """Test conversion of parser results to dict format."""
        loader = RunLoader(str(temp_dir))

        # Sample parser results
        parser_results = [
            {
                "max_concurrency": 50,
                "output_throughput": 2500.0,
                "mean_ttft_ms": 175.0,
                "mean_itl_ms": 20.0,
                "p99_ttft_ms": 300.0,
            },
            {
                "max_concurrency": 100,
                "output_throughput": 5000.0,
                "mean_ttft_ms": 200.0,
                "mean_itl_ms": 22.0,
                "p99_ttft_ms": 350.0,
            },
        ]

        # Convert to dict format
        result_dict = loader._convert_parser_results_to_dict(parser_results)

        # Verify structure
        assert result_dict["concurrencies"] == [50, 100]
        assert result_dict["output_tps"] == [2500.0, 5000.0]
        assert result_dict["mean_ttft_ms"] == [175.0, 200.0]
        assert result_dict["mean_itl_ms"] == [20.0, 22.0]
        assert result_dict["p99_ttft_ms"] == [300.0, 350.0]

    def test_mooncake_router_directory_detection(self, temp_dir, sample_run_metadata):
        """Test that mooncake-router directories are detected correctly."""
        # Create run directory
        run_dir = temp_dir / "12345_2P_4D_20250126_120000"
        run_dir.mkdir()

        # Create metadata with mooncake-router benchmark type
        metadata = sample_run_metadata.copy()
        metadata["benchmark"]["type"] = "mooncake-router"
        metadata_path = run_dir / "12345.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # Create benchmark results directory
        bench_dir = run_dir / "mooncake-router_isl_8192_osl_1024"
        bench_dir.mkdir()

        # Create AIPerf result JSON
        aiperf_data = {
            "output_token_throughput": {"avg": 1150.92},
            "request_throughput": {"avg": 3.37},
            "time_to_first_token": {"avg": 150.5},
            "inter_token_latency": {"avg": 18.5},
            "request_count": {"avg": 1000},
        }
        result_path = bench_dir / "profile_export_aiperf.json"
        with open(result_path, "w") as f:
            json.dump(aiperf_data, f)

        # Load the run
        loader = RunLoader(str(temp_dir))
        run = loader.load_single("12345_2P_4D_20250126_120000")

        # Verify run was loaded
        assert run is not None
        # Verify mooncake-router results were parsed
        assert len(run.profiler.output_tps) >= 1

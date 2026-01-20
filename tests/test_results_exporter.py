# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for results exporter."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from srtctl.core.results_exporter import BenchmarkDatapoint, ResultsExporter


class TestBenchmarkDatapoint:
    """Test BenchmarkDatapoint derived metrics."""

    def test_computed_metrics(self):
        """Derived metrics are computed correctly."""
        dp = BenchmarkDatapoint(
            job_id="12345",
            run_name="test_run",
            run_date="20250116_120000",
            benchmark_type="sa-bench",
            gpu_type="h100",
            gpus_per_node=8,
            prefill_nodes=1,
            decode_nodes=2,
            prefill_workers=1,
            decode_workers=2,
            agg_workers=0,
            total_gpus=24,
            isl=1024,
            osl=1024,
            concurrency=32,
            request_rate=None,
            output_tps=12000.0,
            total_tps=24000.0,
            mean_tpot_ms=10.0,
        )

        # output_tps_per_gpu = 12000 / 24 = 500
        assert dp.output_tps_per_gpu == 500.0

        # total_tps_per_gpu = 24000 / 24 = 1000
        assert dp.total_tps_per_gpu == 1000.0

        # output_tps_per_user = 1000 / 10 = 100
        assert dp.output_tps_per_user == 100.0

    def test_computed_metrics_zero_gpus(self):
        """Handles zero GPUs gracefully."""
        dp = BenchmarkDatapoint(
            job_id="12345",
            run_name="test_run",
            run_date="20250116_120000",
            benchmark_type="sa-bench",
            gpu_type="h100",
            gpus_per_node=8,
            prefill_nodes=0,
            decode_nodes=0,
            prefill_workers=0,
            decode_workers=0,
            agg_workers=0,
            total_gpus=0,
            isl=1024,
            osl=1024,
            concurrency=32,
            request_rate=None,
            output_tps=12000.0,
        )

        # Should not raise, just leave as None
        assert dp.output_tps_per_gpu is None


class TestResultsExporter:
    """Test ResultsExporter."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        config = MagicMock()
        config.benchmark.type = "sa-bench"
        config.benchmark.isl = 1024
        config.benchmark.osl = 1024
        config.profiling.enabled = False
        config.resources.gpu_type = "h100"
        config.resources.gpus_per_node = 8
        config.resources.is_disaggregated = True
        config.resources.prefill_nodes = 1
        config.resources.decode_nodes = 2
        config.resources.num_prefill = 1
        config.resources.num_decode = 2
        config.resources.num_agg = 0
        config.resources.agg_nodes = None
        return config

    @pytest.fixture
    def mock_runtime(self, tmp_path):
        """Create a mock runtime with temp log dir."""
        runtime = MagicMock()
        runtime.job_id = "12345"
        runtime.run_name = "test_run_12345"
        runtime.log_dir = tmp_path
        return runtime

    def test_collect_empty_results(self, mock_config, mock_runtime):
        """Handles empty results directory."""
        exporter = ResultsExporter(config=mock_config, runtime=mock_runtime)
        datapoints = exporter.collect_results()
        assert datapoints == []

    def test_collect_results(self, mock_config, mock_runtime):
        """Collects results from JSON files."""
        # Create a result directory with JSON files
        result_dir = mock_runtime.log_dir / "sa-bench_isl_1024_osl_1024"
        result_dir.mkdir()

        # Create some result JSON files
        for concurrency in [4, 8, 16, 32]:
            result_file = result_dir / f"result_c{concurrency}.json"
            result_file.write_text(
                json.dumps({
                    "max_concurrency": concurrency,
                    "output_throughput": 1000.0 * concurrency,
                    "total_token_throughput": 2000.0 * concurrency,
                    "mean_ttft_ms": 100.0,
                    "mean_tpot_ms": 10.0,
                    "mean_itl_ms": 10.0,
                })
            )

        exporter = ResultsExporter(config=mock_config, runtime=mock_runtime)
        datapoints = exporter.collect_results()

        assert len(datapoints) == 4
        # Should be sorted by concurrency
        assert [dp.concurrency for dp in datapoints] == [4, 8, 16, 32]
        assert datapoints[0].output_tps == 4000.0
        assert datapoints[3].output_tps == 32000.0

    def test_export_csv_and_parquet(self, mock_config, mock_runtime):
        """Exports to both CSV and Parquet formats."""
        # Create a result directory with JSON files
        result_dir = mock_runtime.log_dir / "sa-bench_isl_1024_osl_1024"
        result_dir.mkdir()

        result_file = result_dir / "result_c32.json"
        result_file.write_text(
            json.dumps({
                "max_concurrency": 32,
                "output_throughput": 12000.0,
                "mean_ttft_ms": 100.0,
                "mean_tpot_ms": 10.0,
            })
        )

        exporter = ResultsExporter(config=mock_config, runtime=mock_runtime)
        exported = exporter.export()

        assert "csv" in exported
        assert "parquet" in exported
        assert exported["csv"].exists()
        assert exported["parquet"].exists()
        assert exported["csv"].name == "12345_results.csv"
        assert exported["parquet"].name == "12345_results.parquet"

    def test_to_dataframe(self, mock_config, mock_runtime):
        """Converts to pandas DataFrame."""
        # Create a result directory with JSON files
        result_dir = mock_runtime.log_dir / "sa-bench_isl_1024_osl_1024"
        result_dir.mkdir()

        result_file = result_dir / "result_c32.json"
        result_file.write_text(
            json.dumps({
                "max_concurrency": 32,
                "output_throughput": 12000.0,
                "total_token_throughput": 24000.0,
                "mean_ttft_ms": 100.0,
                "mean_tpot_ms": 10.0,
            })
        )

        exporter = ResultsExporter(config=mock_config, runtime=mock_runtime)
        df = exporter.to_dataframe()

        assert len(df) == 1
        assert df.iloc[0]["job_id"] == "12345"
        assert df.iloc[0]["concurrency"] == 32
        assert df.iloc[0]["output_tps"] == 12000.0
        assert df.iloc[0]["total_tps"] == 24000.0

    def test_get_summary(self, mock_config, mock_runtime):
        """Gets summary statistics."""
        # Create a result directory with JSON files
        result_dir = mock_runtime.log_dir / "sa-bench_isl_1024_osl_1024"
        result_dir.mkdir()

        # Create results with different throughputs
        for concurrency, tps in [(4, 4000.0), (8, 8000.0), (16, 12000.0), (32, 10000.0)]:
            result_file = result_dir / f"result_c{concurrency}.json"
            result_file.write_text(
                json.dumps({
                    "max_concurrency": concurrency,
                    "output_throughput": tps,
                })
            )

        exporter = ResultsExporter(config=mock_config, runtime=mock_runtime)
        exporter.collect_results()
        summary = exporter.get_summary()

        assert summary["num_datapoints"] == 4
        assert summary["peak_output_tps"] == 12000.0
        assert summary["peak_concurrency"] == 16  # Peak is at concurrency 16


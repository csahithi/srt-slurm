# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for benchmark runners."""

import shlex
from unittest.mock import MagicMock, patch

import pytest

from srtctl.benchmarks import get_runner, list_benchmarks
from srtctl.benchmarks.base import SCRIPTS_DIR


class TestBenchmarkRegistry:
    """Test benchmark runner registry."""

    def test_list_benchmarks(self):
        """All expected benchmarks are registered."""
        benchmarks = list_benchmarks()
        assert "sa-bench" in benchmarks
        assert "mmlu" in benchmarks
        assert "gpqa" in benchmarks
        assert "longbenchv2" in benchmarks
        assert "router" in benchmarks
        assert "profiling" in benchmarks

    def test_get_runner_valid(self):
        """Can get runner for valid benchmark type."""
        runner = get_runner("sa-bench")
        assert runner.name == "SA-Bench"
        assert "sa-bench" in runner.script_path

    def test_get_runner_invalid(self):
        """Raises ValueError for unknown benchmark type."""
        with pytest.raises(ValueError, match="Unknown benchmark type"):
            get_runner("nonexistent-benchmark")


class TestSABenchRunner:
    """Test SA-Bench runner."""

    def test_validate_config_missing_isl(self):
        """Validates that isl is required."""
        from srtctl.benchmarks.sa_bench import SABenchRunner
        from srtctl.core.schema import (
            BenchmarkConfig,
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        runner = SABenchRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(type="sa-bench", osl=1024, concurrencies="4x8"),
        )
        errors = runner.validate_config(config)
        assert any("isl" in e for e in errors)

    def test_validate_config_valid(self):
        """Valid config passes validation."""
        from srtctl.benchmarks.sa_bench import SABenchRunner
        from srtctl.core.schema import (
            BenchmarkConfig,
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        runner = SABenchRunner()
        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/image", precision="fp4"),
            resources=ResourceConfig(gpu_type="h100"),
            benchmark=BenchmarkConfig(
                type="sa-bench", isl=1024, osl=1024, concurrencies="4x8"
            ),
        )
        errors = runner.validate_config(config)
        assert errors == []


class TestScriptsExist:
    """Test that benchmark scripts exist."""

    def test_scripts_dir_exists(self):
        """Scripts directory exists."""
        assert SCRIPTS_DIR.exists()

    def test_sa_bench_script_exists(self):
        """SA-Bench script exists."""
        script = SCRIPTS_DIR / "sa-bench" / "bench.sh"
        assert script.exists()

    def test_mmlu_script_exists(self):
        """MMLU script exists."""
        script = SCRIPTS_DIR / "mmlu" / "bench.sh"
        assert script.exists()


class TestBenchmarkCommandLogging:
    """Test that benchmark commands are logged correctly."""

    def test_benchmark_command_logged_to_stdout(self, tmp_path):
        """Verify benchmark command is logged via logger.info."""
        from srtctl.cli.mixins.benchmark_stage import BenchmarkStageMixin

        # Create a mock runner
        mock_runner = MagicMock()
        mock_runner.name = "SA-Bench"
        mock_runner.script_path = "/srtctl-benchmarks/sa-bench/bench.sh"
        mock_runner.build_command.return_value = [
            "bash",
            "/srtctl-benchmarks/sa-bench/bench.sh",
            "http://localhost:8000",
            "1024",
            "1024",
            "4x8x16x32",
            "inf",
        ]

        # Create mock config and runtime
        mock_config = MagicMock()
        mock_config.profiling.enabled = False

        mock_runtime = MagicMock()
        mock_runtime.log_dir = tmp_path
        mock_runtime.nodes.head = "node1"
        mock_runtime.container_image = "/container.sqsh"
        mock_runtime.container_mounts = {}

        # Create a minimal mixin instance
        class TestMixin(BenchmarkStageMixin):
            def __init__(self):
                self.config = mock_config
                self.runtime = mock_runtime

            @property
            def endpoints(self):
                return []

        mixin = TestMixin()

        # Mock the srun process to return immediately
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 0

        with patch("srtctl.cli.mixins.benchmark_stage.start_srun_process", return_value=mock_proc):
            with patch("srtctl.cli.mixins.benchmark_stage.logger") as mock_logger:
                import threading

                stop_event = threading.Event()
                mixin._run_benchmark_script(mock_runner, tmp_path / "benchmark.out", stop_event)

                # Verify logger.info was called with the command
                log_calls = [call for call in mock_logger.info.call_args_list]

                # Find the call that logs the command
                command_logged = False
                expected_cmd = [
                    "bash",
                    "/srtctl-benchmarks/sa-bench/bench.sh",
                    "http://localhost:8000",
                    "1024",
                    "1024",
                    "4x8x16x32",
                    "inf",
                ]
                expected_cmd_str = shlex.join(expected_cmd)

                for call in log_calls:
                    args = call[0]
                    if len(args) >= 2 and "Command:" in str(args[0]):
                        # The format is logger.info("Command: %s", shlex.join(cmd))
                        assert args[1] == expected_cmd_str
                        command_logged = True
                        break

                assert command_logged, f"Expected command to be logged. Log calls: {log_calls}"

    def test_benchmark_script_path_logged(self, tmp_path):
        """Verify benchmark script path is logged."""
        from srtctl.cli.mixins.benchmark_stage import BenchmarkStageMixin

        mock_runner = MagicMock()
        mock_runner.name = "SA-Bench"
        mock_runner.script_path = "/srtctl-benchmarks/sa-bench/bench.sh"
        mock_runner.build_command.return_value = ["bash", mock_runner.script_path]

        mock_config = MagicMock()
        mock_config.profiling.enabled = False

        mock_runtime = MagicMock()
        mock_runtime.log_dir = tmp_path
        mock_runtime.nodes.head = "node1"
        mock_runtime.container_image = "/container.sqsh"
        mock_runtime.container_mounts = {}

        class TestMixin(BenchmarkStageMixin):
            def __init__(self):
                self.config = mock_config
                self.runtime = mock_runtime

            @property
            def endpoints(self):
                return []

        mixin = TestMixin()

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 0

        with patch("srtctl.cli.mixins.benchmark_stage.start_srun_process", return_value=mock_proc):
            with patch("srtctl.cli.mixins.benchmark_stage.logger") as mock_logger:
                import threading

                stop_event = threading.Event()
                mixin._run_benchmark_script(mock_runner, tmp_path / "benchmark.out", stop_event)

                # Verify script path was logged
                script_logged = False
                for call in mock_logger.info.call_args_list:
                    args = call[0]
                    if len(args) >= 2 and "Script:" in str(args[0]):
                        assert args[1] == "/srtctl-benchmarks/sa-bench/bench.sh"
                        script_logged = True
                        break

                assert script_logged, "Expected script path to be logged"

    def test_benchmark_log_file_path_logged(self, tmp_path):
        """Verify benchmark log file path is logged."""
        from srtctl.cli.mixins.benchmark_stage import BenchmarkStageMixin

        mock_runner = MagicMock()
        mock_runner.name = "SA-Bench"
        mock_runner.script_path = "/srtctl-benchmarks/sa-bench/bench.sh"
        mock_runner.build_command.return_value = ["bash", mock_runner.script_path]

        mock_config = MagicMock()
        mock_config.profiling.enabled = False

        mock_runtime = MagicMock()
        mock_runtime.log_dir = tmp_path
        mock_runtime.nodes.head = "node1"
        mock_runtime.container_image = "/container.sqsh"
        mock_runtime.container_mounts = {}

        class TestMixin(BenchmarkStageMixin):
            def __init__(self):
                self.config = mock_config
                self.runtime = mock_runtime

            @property
            def endpoints(self):
                return []

        mixin = TestMixin()

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 0

        log_file = tmp_path / "benchmark.out"

        with patch("srtctl.cli.mixins.benchmark_stage.start_srun_process", return_value=mock_proc):
            with patch("srtctl.cli.mixins.benchmark_stage.logger") as mock_logger:
                import threading

                stop_event = threading.Event()
                mixin._run_benchmark_script(mock_runner, log_file, stop_event)

                # Verify log file path was logged
                log_path_logged = False
                for call in mock_logger.info.call_args_list:
                    args = call[0]
                    if len(args) >= 2 and "Log:" in str(args[0]):
                        assert args[1] == log_file
                        log_path_logged = True
                        break

                assert log_path_logged, "Expected log file path to be logged"


# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for benchmark and node log parsers.

Tests the parsing infrastructure including:
- Parser registry (benchmark and node parsers)
- SA-Bench parser
- Mooncake Router parser
- SGLang node parser
- TRTLLM node parser
"""

import json
import tempfile
from pathlib import Path

import pytest

from analysis.srtlog.parsers import (
    BenchmarkLaunchCommand,
    NodeLaunchCommand,
    get_benchmark_parser,
    get_node_parser,
    list_benchmark_parsers,
    list_node_parsers,
)
from tests.fixtures_parsers import (
    ParserTestHarness,
    SampleMooncakeRouterData,
    SampleSABenchData,
    SampleSGLangLogData,
    SampleTRTLLMLogData,
)


class TestParserRegistry:
    """Test the parser registration system."""

    def test_list_benchmark_parsers(self):
        """Test listing registered benchmark parsers."""
        parsers = list_benchmark_parsers()
        assert "sa-bench" in parsers
        assert "mooncake-router" in parsers
        assert len(parsers) >= 2

    def test_list_node_parsers(self):
        """Test listing registered node parsers."""
        parsers = list_node_parsers()
        assert "sglang" in parsers
        assert "trtllm" in parsers
        assert len(parsers) >= 2

    def test_get_benchmark_parser_sa_bench(self):
        """Test getting SA-Bench parser."""
        parser = get_benchmark_parser("sa-bench")
        assert parser.benchmark_type == "sa-bench"

    def test_get_benchmark_parser_mooncake_router(self):
        """Test getting Mooncake Router parser."""
        parser = get_benchmark_parser("mooncake-router")
        assert parser.benchmark_type == "mooncake-router"

    def test_get_benchmark_parser_invalid(self):
        """Test getting invalid benchmark parser."""
        with pytest.raises(ValueError, match="No benchmark parser registered"):
            get_benchmark_parser("invalid-benchmark")

    def test_get_node_parser_sglang(self):
        """Test getting SGLang parser."""
        parser = get_node_parser("sglang")
        assert parser.backend_type == "sglang"

    def test_get_node_parser_trtllm(self):
        """Test getting TRTLLM parser."""
        parser = get_node_parser("trtllm")
        assert parser.backend_type == "trtllm"

    def test_get_node_parser_invalid(self):
        """Test getting invalid node parser."""
        with pytest.raises(ValueError, match="No node parser registered"):
            get_node_parser("invalid-backend")


class TestSABenchParser:
    """Test SA-Bench benchmark parser."""

    @pytest.fixture
    def parser(self):
        """Get SA-Bench parser instance."""
        return get_benchmark_parser("sa-bench")

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_parser_type(self, parser):
        """Test parser type property."""
        assert parser.benchmark_type == "sa-bench"

    def test_parse_result_json(self, parser, temp_dir):
        """Test parsing SA-Bench result JSON file."""
        # Create sample result JSON
        result_data = {
            "max_concurrency": 100,
            "output_throughput": 5000.5,
            "total_token_throughput": 6000.0,
            "request_throughput": 50.5,
            "mean_ttft_ms": 150.5,
            "mean_tpot_ms": 20.5,
            "mean_itl_ms": 18.5,
            "mean_e2el_ms": 2000.0,
            "p99_ttft_ms": 250.0,
            "p99_tpot_ms": 40.0,
            "p99_itl_ms": 35.0,
            "p99_e2el_ms": 3000.0,
            "total_input_tokens": 100000,
            "total_output_tokens": 50000,
            "completed": 1000,
            "duration": 120.5,
        }

        json_path = temp_dir / "result_c100.json"
        with open(json_path, "w") as f:
            json.dump(result_data, f)

        # Parse the file
        result = parser.parse_result_json(json_path)

        # Verify parsing
        assert result["max_concurrency"] == 100
        assert result["output_throughput"] == 5000.5
        assert result["mean_ttft_ms"] == 150.5
        assert result["p99_ttft_ms"] == 250.0
        assert result["total_input_tokens"] == 100000
        assert result["completed"] == 1000

    def test_parse_result_directory(self, parser, temp_dir):
        """Test parsing multiple result JSON files."""
        # Create multiple result files
        for concurrency in [50, 100, 200]:
            result_data = {
                "max_concurrency": concurrency,
                "output_throughput": concurrency * 50.0,
                "mean_ttft_ms": 150.0,
            }
            json_path = temp_dir / f"result_c{concurrency}.json"
            with open(json_path, "w") as f:
                json.dump(result_data, f)

        # Parse all files
        results = parser.parse_result_directory(temp_dir)

        # Verify results are sorted by concurrency
        assert len(results) == 3
        assert results[0]["max_concurrency"] == 50
        assert results[1]["max_concurrency"] == 100
        assert results[2]["max_concurrency"] == 200

    def test_parse_launch_command_tagged(self, parser):
        """Test parsing SA-Bench command with [CMD] tag."""
        log_content = """
[CMD] python -m sglang.bench_serving --model Qwen/Qwen3-32B --base-url http://localhost:8000 --num-prompts 1000 --request-rate inf --max-concurrency 100 --input-len 8192 --output-len 1024
        """

        cmd = parser.parse_launch_command(log_content)

        assert cmd is not None
        assert cmd.benchmark_type == "sa-bench"
        assert "python -m sglang.bench_serving" in cmd.raw_command
        assert cmd.extra_args["model"] == "Qwen/Qwen3-32B"
        assert cmd.extra_args["base_url"] == "http://localhost:8000"
        assert cmd.extra_args["num_prompts"] == 1000
        assert cmd.extra_args["max_concurrency"] == 100
        assert cmd.extra_args["input_len"] == 8192
        assert cmd.extra_args["output_len"] == 1024

    def test_parse_launch_command_header_format(self, parser):
        """Test parsing SA-Bench config from header format."""
        log_content = """
SA-Bench Config: endpoint=http://localhost:8000; isl=8192; osl=1024; concurrencies=28; req_rate=inf; model=dsr1-fp8
        """

        cmd = parser.parse_launch_command(log_content)

        assert cmd is not None
        assert cmd.benchmark_type == "sa-bench"
        assert cmd.extra_args["base_url"] == "http://localhost:8000"
        assert cmd.extra_args["input_len"] == 8192
        assert cmd.extra_args["output_len"] == 1024
        assert cmd.extra_args["max_concurrency"] == 28
        assert cmd.extra_args["request_rate"] == "inf"
        assert cmd.extra_args["model"] == "dsr1-fp8"

    def test_parse_launch_command_not_found(self, parser):
        """Test parsing when no command is found."""
        log_content = "Some random log content\nNo benchmark commands here\nJust regular logs"
        cmd = parser.parse_launch_command(log_content)
        assert cmd is None


class TestMooncakeRouterParser:
    """Test Mooncake Router benchmark parser."""

    @pytest.fixture
    def parser(self):
        """Get Mooncake Router parser instance."""
        return get_benchmark_parser("mooncake-router")

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_parser_type(self, parser):
        """Test parser type property."""
        assert parser.benchmark_type == "mooncake-router"

    def test_parse_result_json(self, parser, temp_dir):
        """Test parsing AIPerf result JSON file."""
        # Create sample AIPerf JSON
        result_data = {
            "output_token_throughput": {"avg": 1150.92, "p50": 1100.0, "p99": 1200.0, "std": 50.0},
            "request_throughput": {"avg": 3.37, "p50": 3.3, "p99": 3.5, "std": 0.1},
            "time_to_first_token": {"avg": 150.5, "p50": 145.0, "p99": 200.0, "std": 25.0},
            "inter_token_latency": {"avg": 18.5, "p50": 18.0, "p99": 25.0, "std": 3.0},
            "request_latency": {"avg": 2000.0, "p50": 1900.0, "p99": 2500.0, "std": 200.0},
            "request_count": {"avg": 1000},
        }

        json_path = temp_dir / "profile_export_aiperf.json"
        with open(json_path, "w") as f:
            json.dump(result_data, f)

        # Parse the file
        result = parser.parse_result_json(json_path)

        # Verify parsing
        assert result["output_tps"] == 1150.92
        assert result["request_throughput"] == 3.37
        assert result["mean_ttft_ms"] == 150.5
        assert result["median_ttft_ms"] == 145.0
        assert result["p99_ttft_ms"] == 200.0
        assert result["mean_itl_ms"] == 18.5
        assert result["completed"] == 1000

    def test_parse_launch_command_aiperf(self, parser):
        """Test parsing AIPerf command."""
        log_content = """
[CMD] aiperf profile -m "Qwen/Qwen3-32B" --url "http://localhost:8000" --concurrency 10 --request-count 1000
        """

        cmd = parser.parse_launch_command(log_content)

        assert cmd is not None
        assert cmd.benchmark_type == "mooncake-router"
        assert "aiperf" in cmd.raw_command
        assert cmd.extra_args["model"] == "Qwen/Qwen3-32B"
        assert cmd.extra_args["base_url"] == "http://localhost:8000"
        assert cmd.extra_args["max_concurrency"] == 10
        assert cmd.extra_args["num_prompts"] == 1000

    def test_parse_launch_command_header(self, parser):
        """Test parsing from header format."""
        log_content = """
Mooncake Router Benchmark
Endpoint: http://localhost:8000
Model: Qwen/Qwen3-32B
Workload: conversation
        """

        cmd = parser.parse_launch_command(log_content)

        assert cmd is not None
        assert cmd.benchmark_type == "mooncake-router"
        assert cmd.extra_args["base_url"] == "http://localhost:8000"
        assert cmd.extra_args["model"] == "Qwen/Qwen3-32B"
        assert cmd.extra_args["dataset"] == "conversation"


class TestSGLangNodeParser:
    """Test SGLang node log parser."""

    @pytest.fixture
    def parser(self):
        """Get SGLang parser instance."""
        return get_node_parser("sglang")

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_parser_type(self, parser):
        """Test parser type property."""
        assert parser.backend_type == "sglang"

    def test_parse_prefill_batch_line(self, parser):
        """Test parsing prefill batch log line."""
        line = "[2m2025-12-30T15:52:38.206058Z[0m [32m INFO[0m Prefill batch, #new-seq: 5, #new-token: 40960, #cached-token: 0, token usage: 0.78, #running-req: 5, #queue-req: 0, #prealloc-req: 0, #inflight-req: 0, input throughput (token/s): 5000.5"

        metrics = parser._parse_prefill_batch_line(line)

        assert metrics is not None
        assert metrics["type"] == "prefill"
        assert metrics["new_seq"] == 5
        assert metrics["new_token"] == 40960
        assert metrics["cached_token"] == 0
        assert metrics["token_usage"] == 0.78
        assert metrics["running_req"] == 5
        assert metrics["input_throughput"] == 5000.5

    def test_parse_decode_batch_line(self, parser):
        """Test parsing decode batch log line."""
        line = "[2m2025-12-30T15:52:38.206058Z[0m [32m INFO[0m Decode batch, #running-req: 10, #token: 512, token usage: 0.85, pre-allocated usage: 0.10, #prealloc-req: 2, #transfer-req: 0, #queue-req: 0, gen throughput (token/s): 1500.5"

        metrics = parser._parse_decode_batch_line(line)

        assert metrics is not None
        assert metrics["type"] == "decode"
        assert metrics["running_req"] == 10
        assert metrics["num_tokens"] == 512
        assert metrics["token_usage"] == 0.85
        assert metrics["preallocated_usage"] == 0.10
        assert metrics["gen_throughput"] == 1500.5

    def test_parse_memory_line(self, parser):
        """Test parsing memory log line."""
        line = "[2m2025-12-30T15:52:38.206058Z[0m [32m INFO[0m avail mem=75.11 GB, mem usage=107.07 GB, KV size: 17.16 GB, #tokens: 524288"

        metrics = parser._parse_memory_line(line)

        assert metrics is not None
        # This line has KV size, so it should be marked as kv_cache type
        assert metrics["type"] == "kv_cache"
        assert metrics["avail_mem_gb"] == 75.11
        assert metrics["mem_usage_gb"] == 107.07
        assert metrics["kv_cache_gb"] == 17.16
        assert metrics["kv_tokens"] == 524288

    def test_parse_memory_line_without_kv(self, parser):
        """Test parsing memory log line without KV info."""
        line = "[2m2025-12-30T15:52:38.206058Z[0m [32m INFO[0m avail mem=75.11 GB, mem usage=107.07 GB"

        metrics = parser._parse_memory_line(line)

        assert metrics is not None
        assert metrics["type"] == "memory"
        assert metrics["avail_mem_gb"] == 75.11
        assert metrics["mem_usage_gb"] == 107.07

    def test_parse_single_log(self, parser, temp_dir):
        """Test parsing a complete SGLang log file."""
        log_content = """
[2m2025-12-30T15:52:38.206058Z[0m [32m INFO[0m Starting SGLang server with server_args=ServerArgs(tp_size=8, dp_size=1, ep_size=1, model_path=/models/qwen3-32b)
[2m2025-12-30T15:52:40.206058Z[0m [32m INFO[0m Prefill batch, #new-seq: 5, #new-token: 40960, #cached-token: 0, token usage: 0.78, #running-req: 5, #queue-req: 0, #prealloc-req: 0, #inflight-req: 0, input throughput (token/s): 5000.5
[2m2025-12-30T15:52:41.206058Z[0m [32m INFO[0m Decode batch, #running-req: 5, #token: 512, token usage: 0.85, gen throughput (token/s): 1500.5
[2m2025-12-30T15:52:42.206058Z[0m [32m INFO[0m avail mem=75.11 GB, mem usage=107.07 GB
        """

        log_path = temp_dir / "eos0219_prefill_w0.out"
        log_path.write_text(log_content)

        node = parser.parse_single_log(log_path)

        assert node is not None
        assert node.node_name == "eos0219"
        assert node.worker_type == "prefill"
        assert node.worker_id == "w0"
        assert len(node.batches) == 2  # 1 prefill + 1 decode
        assert len(node.memory_snapshots) == 1
        assert node.config["tp_size"] == 8
        assert node.config["dp_size"] == 1
        assert node.config["ep_size"] == 1

    def test_parse_launch_command(self, parser):
        """Test parsing SGLang launch command."""
        log_content = """
[CMD] python -m sglang.launch_server --model /models/qwen3-32b --tp-size 8 --dp-size 1 --host 10.0.0.1 --port 30000 --max-num-seqs 1024 --disaggregation-mode prefill
        """

        cmd = parser.parse_launch_command(log_content, "prefill")

        assert cmd is not None
        assert cmd.backend_type == "sglang"
        assert cmd.worker_type == "prefill"
        assert cmd.extra_args["model_path"] == "/models/qwen3-32b"
        assert cmd.extra_args["tp_size"] == 8
        assert cmd.extra_args["dp_size"] == 1
        assert cmd.extra_args["host"] == "10.0.0.1"
        assert cmd.extra_args["port"] == 30000
        assert cmd.extra_args["max_num_seqs"] == 1024
        assert cmd.extra_args["disaggregation_mode"] == "prefill"

    def test_extract_node_info_from_filename(self, parser):
        """Test extracting node info from filename."""
        result = parser._extract_node_info_from_filename("eos0219_prefill_w0.out")

        assert result is not None
        assert result["node"] == "eos0219"
        assert result["worker_type"] == "prefill"
        assert result["worker_id"] == "w0"


class TestTRTLLMNodeParser:
    """Test TRTLLM node log parser."""

    @pytest.fixture
    def parser(self):
        """Get TRTLLM parser instance."""
        return get_node_parser("trtllm")

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_parser_type(self, parser):
        """Test parser type property."""
        assert parser.backend_type == "trtllm"

    def test_parse_iteration_logs(self, parser):
        """Test parsing TRTLLM iteration logs."""
        log_content = """
[01/16/2026-06:20:17] [TRT-LLM] [RANK 0] [I] iter = 5559, host_step_time = 50.5ms, num_scheduled_requests: 3, states = {'num_ctx_requests': 2, 'num_ctx_tokens': 16384, 'num_generation_tokens': 0}
[01/16/2026-06:20:18] [TRT-LLM] [RANK 0] [I] iter = 5560, host_step_time = 20.0ms, num_scheduled_requests: 5, states = {'num_ctx_requests': 0, 'num_ctx_tokens': 0, 'num_generation_tokens': 512}
        """

        batches = parser._parse_iteration_logs(log_content, "prefill")

        assert len(batches) == 2

        # First batch (prefill)
        assert batches[0].batch_type == "prefill"
        assert batches[0].running_req == 3
        assert batches[0].new_token == 16384
        assert batches[0].input_throughput is not None
        assert batches[0].input_throughput > 300000  # 16384 * 1000 / 50.5

        # Second batch (decode)
        assert batches[1].batch_type == "decode"
        assert batches[1].running_req == 5
        assert batches[1].num_tokens == 512
        assert batches[1].gen_throughput is not None
        assert batches[1].gen_throughput > 25000  # 512 * 1000 / 20

    def test_parse_memory_info(self, parser):
        """Test parsing TRTLLM memory information."""
        log_content = """
[01/16/2026-06:20:17] [TRT-LLM] [I] Peak memory during memory usage profiling (torch + non-torch): 91.46 GiB, available KV cache memory when calculating max tokens: 41.11 GiB, fraction is set 0.85, kv size is 35136. device total memory 139.81 GiB
[MemUsageChange] Allocated 41.11 GiB for max tokens (524288)
        """

        memory_snapshots = parser._parse_memory_info(log_content)

        assert len(memory_snapshots) == 2

        # First snapshot (peak memory)
        assert memory_snapshots[0].metric_type == "memory"
        assert memory_snapshots[0].mem_usage_gb == 91.46
        assert memory_snapshots[0].kv_cache_gb == 41.11
        assert memory_snapshots[0].avail_mem_gb > 48  # 139.81 - 91.46

        # Second snapshot (KV allocation)
        assert memory_snapshots[1].metric_type == "kv_cache"
        assert memory_snapshots[1].kv_cache_gb == 41.11
        assert memory_snapshots[1].kv_tokens == 524288

    def test_parse_single_log(self, parser, temp_dir):
        """Test parsing a complete TRTLLM log file."""
        log_content = """
[33mRank0 run python3 -m dynamo.trtllm --model-path /model --served-model-name dsr1-fp8 --disaggregation-mode prefill[0m
Initializing the worker with config: Config(tensor_parallel_size=8, pipeline_parallel_size=1, expert_parallel_size=1, max_batch_size=256)
TensorRT-LLM engine args: {'tensor_parallel_size': 8, 'pipeline_parallel_size': 1, 'moe_expert_parallel_size': 1, 'max_batch_size': 256, 'max_seq_len': 131072}
[01/16/2026-06:20:17] [TRT-LLM] [RANK 0] [I] iter = 5559, num_scheduled_requests: 3, states = {'num_ctx_requests': 2, 'num_ctx_tokens': 16384, 'num_generation_tokens': 0}
        """

        log_path = temp_dir / "worker-0_prefill_w0.out"
        log_path.write_text(log_content)

        node = parser.parse_single_log(log_path)

        assert node is not None
        assert node.node_name == "worker-0"
        assert node.worker_type == "prefill"
        assert node.worker_id == "w0"
        assert len(node.batches) == 1
        assert node.config["tp_size"] == 8
        assert node.config["pp_size"] == 1
        assert node.config["ep_size"] == 1
        assert node.config["max_batch_size"] == 256
        assert node.config["max_seq_len"] == 131072

    def test_parse_launch_command(self, parser):
        """Test parsing TRTLLM launch command."""
        log_content = """
[CMD] python3 -m dynamo.trtllm --model-path /models/qwen3-32b --served-model-name dsr1-fp8 --disaggregation-mode prefill --host 10.0.0.1 --port 30000
TensorRT-LLM engine args: {'tensor_parallel_size': 8, 'pipeline_parallel_size': 1, 'max_batch_size': 256, 'max_seq_len': 131072}
        """

        cmd = parser.parse_launch_command(log_content, "prefill")

        assert cmd is not None
        assert cmd.backend_type == "trtllm"
        assert cmd.worker_type == "prefill"
        assert cmd.extra_args["model_path"] == "/models/qwen3-32b"
        assert cmd.extra_args["served_model_name"] == "dsr1-fp8"
        assert cmd.extra_args["disaggregation_mode"] == "prefill"
        assert cmd.extra_args["host"] == "10.0.0.1"
        assert cmd.extra_args["port"] == 30000
        assert cmd.extra_args["tp_size"] == 8
        assert cmd.extra_args["pp_size"] == 1
        assert cmd.extra_args["max_num_seqs"] == 256
        assert cmd.extra_args["max_model_len"] == 131072

    def test_extract_node_info_from_filename(self, parser):
        """Test extracting node info from filename."""
        result = parser._extract_node_info_from_filename("worker-0_decode_w1.err")

        assert result is not None
        assert result["node"] == "worker-0"
        assert result["worker_type"] == "decode"
        assert result["worker_id"] == "w1"


class TestBenchmarkLaunchCommand:
    """Test BenchmarkLaunchCommand dataclass."""

    def test_create_benchmark_launch_command(self):
        """Test creating BenchmarkLaunchCommand."""
        cmd = BenchmarkLaunchCommand(
            benchmark_type="sa-bench",
            raw_command="python -m sglang.bench_serving --model test",
            extra_args={"model": "test", "num_prompts": 1000},
        )

        assert cmd.benchmark_type == "sa-bench"
        assert "sglang.bench_serving" in cmd.raw_command
        assert cmd.extra_args["model"] == "test"
        assert cmd.extra_args["num_prompts"] == 1000


class TestNodeLaunchCommand:
    """Test NodeLaunchCommand dataclass."""

    def test_create_node_launch_command(self):
        """Test creating NodeLaunchCommand."""
        cmd = NodeLaunchCommand(
            backend_type="sglang",
            worker_type="prefill",
            raw_command="python -m sglang.launch_server --model test",
            extra_args={"model_path": "test", "tp_size": 8},
        )

        assert cmd.backend_type == "sglang"
        assert cmd.worker_type == "prefill"
        assert "sglang.launch_server" in cmd.raw_command
        assert cmd.extra_args["model_path"] == "test"
        assert cmd.extra_args["tp_size"] == 8


class TestParserIntegration:
    """Integration tests for parser workflows."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_parse_complete_sa_bench_run(self, temp_dir):
        """Test parsing a complete SA-Bench run with multiple concurrencies."""
        parser = get_benchmark_parser("sa-bench")

        # Use test harness to create run directory
        run_dir = ParserTestHarness.create_sa_bench_run(temp_dir, concurrencies=[50, 100, 200])

        # Parse all results
        results = parser.parse_result_directory(run_dir)

        assert len(results) == 3
        # Verify it's sorted by concurrency
        assert [r["max_concurrency"] for r in results] == [50, 100, 200]
        # Verify throughput scales with concurrency
        assert results[0]["output_throughput"] == 2500.0
        assert results[1]["output_throughput"] == 5000.0
        assert results[2]["output_throughput"] == 10000.0

        # Verify using harness utility
        for result in results:
            ParserTestHarness.assert_valid_benchmark_results(
                result,
                expected_fields=[
                    "output_throughput",
                    "mean_ttft_ms",
                    "mean_itl_ms",
                    "p99_ttft_ms",
                ],
            )

    def test_parse_mooncake_router_run(self, temp_dir):
        """Test parsing a complete Mooncake Router run."""
        parser = get_benchmark_parser("mooncake-router")

        # Use test harness to create run directory
        run_dir = ParserTestHarness.create_mooncake_router_run(temp_dir)

        # Find and parse AIPerf results
        aiperf_files = parser.find_aiperf_results(run_dir)
        assert len(aiperf_files) == 1

        result = parser.parse_result_json(aiperf_files[0])
        ParserTestHarness.assert_valid_benchmark_results(
            result,
            expected_fields=["output_tps", "request_throughput", "mean_ttft_ms"],
        )

    def test_parse_sglang_node_logs_multiple_workers(self, temp_dir):
        """Test parsing multiple SGLang node log files."""
        parser = get_node_parser("sglang")

        # Use test harness to create log directory
        log_dir = ParserTestHarness.create_sglang_node_logs(temp_dir, num_prefill=2, num_decode=4)

        # Parse all logs
        nodes = parser.parse_logs(log_dir)

        assert len(nodes) == 6  # 2 prefill + 4 decode
        worker_types = [node.worker_type for node in nodes]
        assert worker_types.count("prefill") == 2
        assert worker_types.count("decode") == 4

        # Verify each node
        for node in nodes:
            ParserTestHarness.assert_valid_node_metrics(node, min_batches=1)

    def test_parse_trtllm_node_logs_multiple_workers(self, temp_dir):
        """Test parsing multiple TRTLLM node log files."""
        parser = get_node_parser("trtllm")

        # Use test harness to create log directory
        log_dir = ParserTestHarness.create_trtllm_node_logs(temp_dir, num_prefill=2, num_decode=4)

        # Parse all logs
        nodes = parser.parse_logs(log_dir)

        assert len(nodes) == 6  # 2 prefill + 4 decode
        worker_types = [node.worker_type for node in nodes]
        assert worker_types.count("prefill") == 2
        assert worker_types.count("decode") == 4

        # Verify each node has config
        for node in nodes:
            ParserTestHarness.assert_valid_node_metrics(node, min_batches=1)
            assert "tp_size" in node.config or "max_batch_size" in node.config


class TestParserWithFixtures:
    """Tests using sample data fixtures."""

    def test_sa_bench_sample_data(self):
        """Test SA-Bench parser with sample data."""
        parser = get_benchmark_parser("sa-bench")

        # Parse launch command from sample
        log_content = SampleSABenchData.benchmark_out_content()
        cmd = parser.parse_launch_command(log_content)

        assert cmd is not None
        assert cmd.benchmark_type == "sa-bench"
        assert "model" in cmd.extra_args
        assert "base_url" in cmd.extra_args

    def test_mooncake_router_sample_data(self):
        """Test Mooncake Router parser with sample data."""
        parser = get_benchmark_parser("mooncake-router")

        # Parse launch command from sample
        log_content = SampleMooncakeRouterData.benchmark_out_content()
        cmd = parser.parse_launch_command(log_content)

        assert cmd is not None
        assert cmd.benchmark_type == "mooncake-router"

    def test_sglang_prefill_sample_data(self):
        """Test SGLang parser with prefill sample data."""
        parser = get_node_parser("sglang")

        # Parse launch command from sample
        log_content = SampleSGLangLogData.prefill_log_content()
        cmd = parser.parse_launch_command(log_content, "prefill")

        assert cmd is not None
        assert cmd.backend_type == "sglang"
        assert cmd.worker_type == "prefill"
        assert "tp_size" in cmd.extra_args
        assert cmd.extra_args["tp_size"] == 8

    def test_sglang_decode_sample_data(self):
        """Test SGLang parser with decode sample data."""
        parser = get_node_parser("sglang")

        # Parse launch command from sample
        log_content = SampleSGLangLogData.decode_log_content()
        cmd = parser.parse_launch_command(log_content, "decode")

        assert cmd is not None
        assert cmd.backend_type == "sglang"
        assert cmd.worker_type == "decode"
        assert "tp_size" in cmd.extra_args

    def test_trtllm_prefill_sample_data(self):
        """Test TRTLLM parser with prefill sample data."""
        parser = get_node_parser("trtllm")

        # Parse launch command from sample
        log_content = SampleTRTLLMLogData.prefill_log_content()
        cmd = parser.parse_launch_command(log_content, "prefill")

        assert cmd is not None
        assert cmd.backend_type == "trtllm"
        assert cmd.worker_type == "prefill"
        assert "disaggregation_mode" in cmd.extra_args
        assert cmd.extra_args["disaggregation_mode"] == "prefill"

    def test_trtllm_decode_sample_data(self):
        """Test TRTLLM parser with decode sample data."""
        parser = get_node_parser("trtllm")

        # Parse launch command from sample
        log_content = SampleTRTLLMLogData.decode_log_content()
        cmd = parser.parse_launch_command(log_content, "decode")

        assert cmd is not None
        assert cmd.backend_type == "trtllm"
        assert cmd.worker_type == "decode"


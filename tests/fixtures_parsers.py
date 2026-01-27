# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test fixtures and sample data for parser tests.

Provides reusable test data, log samples, and utilities for testing parsers.
"""

import json
from pathlib import Path
from typing import Any


class SampleSABenchData:
    """Sample data for SA-Bench parser testing."""

    @staticmethod
    def benchmark_out_content() -> str:
        """Sample benchmark.out content."""
        return """
SA-Bench Config: endpoint=http://localhost:8000; isl=8192; osl=1024; concurrencies=50x100x200; req_rate=inf; model=Qwen/Qwen3-32B

[CMD] python -m sglang.bench_serving --model Qwen/Qwen3-32B --base-url http://localhost:8000 --num-prompts 1000 --request-rate inf --max-concurrency 50 --random-input-len 8192 --random-output-len 1024

Starting benchmark run...
Concurrency: 50, Throughput: 2500.5 tok/s, TTFT: 150.5ms, ITL: 20.0ms
Concurrency: 100, Throughput: 5000.0 tok/s, TTFT: 180.0ms, ITL: 22.0ms
Concurrency: 200, Throughput: 9500.5 tok/s, TTFT: 250.0ms, ITL: 25.0ms
Benchmark complete.
        """

    @staticmethod
    def result_json(concurrency: int = 100) -> dict[str, Any]:
        """Sample result JSON data."""
        return {
            "max_concurrency": concurrency,
            "output_throughput": concurrency * 50.0,
            "total_token_throughput": concurrency * 60.0,
            "request_throughput": concurrency * 0.5,
            "request_goodput": concurrency * 0.48,
            "request_rate": float("inf"),
            # Mean latencies
            "mean_ttft_ms": 150.0 + concurrency * 0.5,
            "mean_tpot_ms": 20.0 + concurrency * 0.1,
            "mean_itl_ms": 18.0 + concurrency * 0.08,
            "mean_e2el_ms": 2000.0 + concurrency * 5.0,
            # Median latencies
            "median_ttft_ms": 140.0 + concurrency * 0.45,
            "median_tpot_ms": 19.0 + concurrency * 0.09,
            "median_itl_ms": 17.0 + concurrency * 0.07,
            "median_e2el_ms": 1900.0 + concurrency * 4.5,
            # P99 latencies
            "p99_ttft_ms": 250.0 + concurrency * 1.0,
            "p99_tpot_ms": 40.0 + concurrency * 0.2,
            "p99_itl_ms": 35.0 + concurrency * 0.15,
            "p99_e2el_ms": 3000.0 + concurrency * 10.0,
            # Std dev
            "std_ttft_ms": 25.0,
            "std_tpot_ms": 5.0,
            "std_itl_ms": 3.0,
            "std_e2el_ms": 200.0,
            # Token counts
            "total_input_tokens": concurrency * 8192,
            "total_output_tokens": concurrency * 1024,
            # Metadata
            "duration": 120.5,
            "completed": concurrency * 10,
            "num_prompts": concurrency * 10,
        }


class SampleMooncakeRouterData:
    """Sample data for Mooncake Router parser testing."""

    @staticmethod
    def benchmark_out_content() -> str:
        """Sample benchmark.out content."""
        return """
Mooncake Router Benchmark
Endpoint: http://localhost:8000
Model: Qwen/Qwen3-32B
Workload: conversation

[CMD] aiperf profile -m "Qwen/Qwen3-32B" --url "http://localhost:8000" --concurrency 10 --synthetic-input-tokens-mean 8192 --output-tokens-mean 1024

Starting benchmark...
Request throughput: 3.37 req/s
Output token throughput: 1150.92 tok/s
Time to first token: 150.5 ms
Inter-token latency: 18.5 ms
        """

    @staticmethod
    def aiperf_result_json() -> dict[str, Any]:
        """Sample AIPerf result JSON data."""
        return {
            "output_token_throughput": {
                "avg": 1150.92,
                "p50": 1100.0,
                "p99": 1200.0,
                "std": 50.0,
            },
            "request_throughput": {"avg": 3.37, "p50": 3.3, "p99": 3.5, "std": 0.1},
            "time_to_first_token": {
                "avg": 150.5,
                "p50": 145.0,
                "p99": 200.0,
                "std": 25.0,
            },
            "inter_token_latency": {
                "avg": 18.5,
                "p50": 18.0,
                "p99": 25.0,
                "std": 3.0,
            },
            "request_latency": {
                "avg": 2000.0,
                "p50": 1900.0,
                "p99": 2500.0,
                "std": 200.0,
            },
            "request_count": {"avg": 1000},
            "output_token_throughput_per_user": {"avg": 115.09},
        }


class SampleSGLangLogData:
    """Sample data for SGLang node parser testing."""

    @staticmethod
    def prefill_log_content() -> str:
        """Sample prefill worker log."""
        return """
[2025-12-30 15:52:38 DP0 TP0 EP0] INFO sglang Starting SGLang prefill worker
[2025-12-30 15:52:38 DP0 TP0 EP0] INFO sglang server_args=ServerArgs(tp_size=8, dp_size=1, ep_size=1, model_path=/models/qwen3-32b, served_model_name=Qwen3-32B, host=10.0.0.1, port=30000, disaggregation_mode=prefill, context_length=131072, max_running_requests=1024, mem_fraction_static=0.85, kv_cache_dtype=fp8_e5m2)

[CMD] python -m sglang.launch_server --model /models/qwen3-32b --served-model-name Qwen3-32B --tp-size 8 --dp-size 1 --ep-size 1 --host 10.0.0.1 --port 30000 --disaggregation-mode prefill --context-length 131072 --max-running-requests 1024 --mem-fraction-static 0.85 --kv-cache-dtype fp8_e5m2

[2025-12-30 15:52:40 DP0 TP0 EP0] INFO sglang Prefill batch, #new-seq: 8, #new-token: 65536, #cached-token: 0, token usage: 0.78, #running-req: 8, #queue-req: 0, #prealloc-req: 0, #inflight-req: 0, input throughput (token/s): 6500.5
[2025-12-30 15:52:40 DP0 TP0 EP0] INFO sglang Prefill batch, #new-seq: 5, #new-token: 40960, #cached-token: 0, token usage: 0.85, #running-req: 13, #queue-req: 0, #prealloc-req: 0, #inflight-req: 0, input throughput (token/s): 5120.0
[2025-12-30 15:52:41 DP0 TP0 EP0] INFO sglang Prefill batch, #new-seq: 10, #new-token: 81920, #cached-token: 16384, token usage: 0.90, #running-req: 23, #queue-req: 2, #prealloc-req: 0, #inflight-req: 0, input throughput (token/s): 8192.0
[2025-12-30 15:52:42 DP0 TP0 EP0] INFO sglang avail mem=75.11 GB, mem usage=107.07 GB
[2025-12-30 15:52:43 DP0 TP0 EP0] INFO sglang KV size: 32.50 GB, #tokens: 1048576
        """

    @staticmethod
    def decode_log_content() -> str:
        """Sample decode worker log."""
        return """
[2025-12-30 15:52:38 DP0 TP0 EP0] INFO sglang Starting SGLang decode worker
[2025-12-30 15:52:38 DP0 TP0 EP0] INFO sglang server_args=ServerArgs(tp_size=4, dp_size=1, ep_size=1, model_path=/models/qwen3-32b, disaggregation_mode=decode)

[CMD] python -m sglang.launch_server --model /models/qwen3-32b --tp-size 4 --disaggregation-mode decode

[2025-12-30 15:52:40 DP0 TP0 EP0] INFO sglang Decode batch, #running-req: 15, #token: 512, token usage: 0.65, pre-allocated usage: 0.10, #prealloc-req: 3, #transfer-req: 0, #queue-req: 0, gen throughput (token/s): 2048.0
[2025-12-30 15:52:40 DP0 TP0 EP0] INFO sglang Decode batch, #running-req: 20, #token: 768, token usage: 0.72, pre-allocated usage: 0.15, #prealloc-req: 5, #transfer-req: 2, #queue-req: 0, gen throughput (token/s): 3072.0
[2025-12-30 15:52:41 DP0 TP0 EP0] INFO sglang Decode batch, #running-req: 18, #token: 640, token usage: 0.70, pre-allocated usage: 0.12, #prealloc-req: 4, #transfer-req: 1, #queue-req: 0, gen throughput (token/s): 2560.0
[2025-12-30 15:52:42 DP0 TP0 EP0] INFO sglang avail mem=85.00 GB, mem usage=97.00 GB
[2025-12-30 15:52:43 DP0 TP0 EP0] INFO sglang KV size: 48.00 GB, #tokens: 2097152
        """


class SampleTRTLLMLogData:
    """Sample data for TRTLLM node parser testing."""

    @staticmethod
    def prefill_log_content() -> str:
        """Sample TRTLLM prefill worker log."""
        return """
[33mRank0 run python3 -m dynamo.trtllm --model-path /models/qwen3-32b --served-model-name Qwen3-32B-fp8 --disaggregation-mode prefill --host 10.0.0.1 --port 30000[0m

[CMD] python3 -m dynamo.trtllm --model-path /models/qwen3-32b --served-model-name Qwen3-32B-fp8 --disaggregation-mode prefill --host 10.0.0.1 --port 30000

Initializing the worker with config: Config(namespace=dynamo, component=prefill, tensor_parallel_size=8, pipeline_parallel_size=1, expert_parallel_size=1, max_batch_size=256, max_num_tokens=16384, max_seq_len=131072)

TensorRT-LLM engine args: {'tensor_parallel_size': 8, 'pipeline_parallel_size': 1, 'moe_expert_parallel_size': 1, 'max_batch_size': 256, 'max_num_tokens': 16384, 'max_seq_len': 131072}

[01/16/2026-06:20:15] [TRT-LLM] [I] Peak memory during memory usage profiling (torch + non-torch): 91.46 GiB, available KV cache memory when calculating max tokens: 41.11 GiB, fraction is set 0.85, kv size is 35136. device total memory 139.81 GiB

[MemUsageChange] Allocated 41.11 GiB for max tokens (524288)

[01/16/2026-06:20:17] [TRT-LLM] [RANK 0] [I] iter = 5559, host_step_time = 62.5ms, num_scheduled_requests: 5, states = {'num_ctx_requests': 5, 'num_ctx_tokens': 40960, 'num_generation_tokens': 0}
[01/16/2026-06:20:17] [TRT-LLM] [RANK 0] [I] iter = 5560, host_step_time = 80.0ms, num_scheduled_requests: 8, states = {'num_ctx_requests': 8, 'num_ctx_tokens': 65536, 'num_generation_tokens': 0}
[01/16/2026-06:20:18] [TRT-LLM] [RANK 0] [I] iter = 5561, host_step_time = 100.0ms, num_scheduled_requests: 10, states = {'num_ctx_requests': 10, 'num_ctx_tokens': 81920, 'num_generation_tokens': 0}
        """

    @staticmethod
    def decode_log_content() -> str:
        """Sample TRTLLM decode worker log."""
        return """
[33mRank0 run python3 -m dynamo.trtllm --model-path /models/qwen3-32b --served-model-name Qwen3-32B-fp8 --disaggregation-mode decode --host 10.0.0.2 --port 30001[0m

[CMD] python3 -m dynamo.trtllm --model-path /models/qwen3-32b --served-model-name Qwen3-32B-fp8 --disaggregation-mode decode --host 10.0.0.2 --port 30001

Initializing the worker with config: Config(tensor_parallel_size=4, pipeline_parallel_size=1, max_batch_size=512)

TensorRT-LLM engine args: {'tensor_parallel_size': 4, 'pipeline_parallel_size': 1, 'max_batch_size': 512, 'max_seq_len': 131072}

[01/16/2026-06:20:15] [TRT-LLM] [I] Peak memory during memory usage profiling (torch + non-torch): 75.50 GiB, available KV cache memory when calculating max tokens: 55.00 GiB, fraction is set 0.85, kv size is 45000. device total memory 139.81 GiB

[01/16/2026-06:20:17] [TRT-LLM] [RANK 0] [I] iter = 1000, host_step_time = 40.0ms, num_scheduled_requests: 20, states = {'num_ctx_requests': 0, 'num_ctx_tokens': 0, 'num_generation_tokens': 1024}
[01/16/2026-06:20:17] [TRT-LLM] [RANK 0] [I] iter = 1001, host_step_time = 50.0ms, num_scheduled_requests: 25, states = {'num_ctx_requests': 0, 'num_ctx_tokens': 0, 'num_generation_tokens': 1280}
[01/16/2026-06:20:18] [TRT-LLM] [RANK 0] [I] iter = 1002, host_step_time = 45.0ms, num_scheduled_requests: 22, states = {'num_ctx_requests': 0, 'num_ctx_tokens': 0, 'num_generation_tokens': 1152}
        """


class ParserTestHarness:
    """Test harness utilities for parser testing."""

    @staticmethod
    def create_sa_bench_run(temp_dir: Path, concurrencies: list[int] | None = None) -> Path:
        """Create a complete SA-Bench run directory with result files.

        Args:
            temp_dir: Temporary directory to create files in
            concurrencies: List of concurrency levels to create (default: [50, 100, 200])

        Returns:
            Path to the run directory
        """
        if concurrencies is None:
            concurrencies = [50, 100, 200]

        run_dir = temp_dir / "sa_bench_run"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create benchmark.out
        benchmark_out = run_dir / "benchmark.out"
        benchmark_out.write_text(SampleSABenchData.benchmark_out_content())

        # Create result JSON files
        for concurrency in concurrencies:
            result_json = run_dir / f"result_c{concurrency}.json"
            with open(result_json, "w") as f:
                json.dump(SampleSABenchData.result_json(concurrency), f, indent=2)

        return run_dir

    @staticmethod
    def create_mooncake_router_run(temp_dir: Path) -> Path:
        """Create a Mooncake Router run directory with result file.

        Args:
            temp_dir: Temporary directory to create files in

        Returns:
            Path to the run directory
        """
        run_dir = temp_dir / "mooncake_router_run"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create benchmark.out
        benchmark_out = run_dir / "benchmark.out"
        benchmark_out.write_text(SampleMooncakeRouterData.benchmark_out_content())

        # Create AIPerf result JSON
        aiperf_json = run_dir / "profile_export_aiperf.json"
        with open(aiperf_json, "w") as f:
            json.dump(SampleMooncakeRouterData.aiperf_result_json(), f, indent=2)

        return run_dir

    @staticmethod
    def create_sglang_node_logs(
        temp_dir: Path,
        num_prefill: int = 2,
        num_decode: int = 4,
    ) -> Path:
        """Create SGLang node log directory with worker logs.

        Args:
            temp_dir: Temporary directory to create files in
            num_prefill: Number of prefill workers
            num_decode: Number of decode workers

        Returns:
            Path to the log directory
        """
        log_dir = temp_dir
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create prefill worker logs
        for i in range(num_prefill):
            log_file = log_dir / f"node{i:02d}_prefill_w{i}.out"
            log_file.write_text(SampleSGLangLogData.prefill_log_content())

        # Create decode worker logs
        for i in range(num_decode):
            log_file = log_dir / f"node{i+10:02d}_decode_w{i}.out"
            log_file.write_text(SampleSGLangLogData.decode_log_content())

        return log_dir

    @staticmethod
    def create_trtllm_node_logs(
        temp_dir: Path,
        num_prefill: int = 2,
        num_decode: int = 4,
    ) -> Path:
        """Create TRTLLM node log directory with worker logs.

        Args:
            temp_dir: Temporary directory to create files in
            num_prefill: Number of prefill workers
            num_decode: Number of decode workers

        Returns:
            Path to the log directory
        """
        log_dir = temp_dir
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create prefill worker logs
        for i in range(num_prefill):
            log_file = log_dir / f"worker-{i}_prefill_w{i}.out"
            log_file.write_text(SampleTRTLLMLogData.prefill_log_content())

        # Create decode worker logs
        for i in range(num_decode):
            log_file = log_dir / f"worker-{i+10}_decode_w{i}.out"
            log_file.write_text(SampleTRTLLMLogData.decode_log_content())

        return log_dir

    @staticmethod
    def assert_valid_benchmark_results(results: dict, expected_fields: list[str] | None = None):
        """Assert that benchmark results contain valid data.

        Args:
            results: Benchmark results dictionary
            expected_fields: List of fields that must be present (optional)
        """
        if expected_fields is None:
            expected_fields = [
                "output_tps",
                "mean_ttft_ms",
                "mean_itl_ms",
            ]

        for field in expected_fields:
            assert field in results, f"Missing expected field: {field}"
            value = results[field]
            # Check it's not None and if it's a list, check it's not empty
            assert value is not None and (not isinstance(value, list) or len(value) > 0), \
                f"Field {field} is None or empty list"

    @staticmethod
    def assert_valid_node_metrics(node_metrics, min_batches: int = 0):
        """Assert that node metrics are valid.

        Args:
            node_metrics: NodeMetrics object
            min_batches: Minimum number of batches expected
        """
        assert node_metrics is not None
        assert node_metrics.node_name
        assert node_metrics.worker_type
        assert node_metrics.worker_id
        assert len(node_metrics.batches) >= min_batches
        assert isinstance(node_metrics.config, dict)


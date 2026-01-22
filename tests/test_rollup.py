# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the rollup stage mixin."""

import json
from pathlib import Path

import pytest

from srtctl.cli.mixins.rollup_stage import (
    NodeRollup,
    NodesSummary,
    RollupResult,
    RollupStageMixin,
    RollupSummary,
)


class TestNodeRollup:
    """Tests for NodeRollup dataclass."""

    def test_minimal_node_rollup(self):
        """Test creating a NodeRollup with minimal fields."""
        node = NodeRollup(
            node_name="node-01",
            worker_type="prefill",
            worker_id="w0",
        )
        assert node.node_name == "node-01"
        assert node.worker_type == "prefill"
        assert node.worker_id == "w0"
        assert node.total_batches == 0
        assert node.tp_size is None

    def test_prefill_node_rollup(self):
        """Test creating a prefill NodeRollup with all metrics."""
        node = NodeRollup(
            node_name="node-01",
            worker_type="prefill",
            worker_id="w0",
            tp_size=8,
            dp_size=1,
            ep_size=1,
            avail_mem_gb=75.0,
            mem_usage_gb=107.0,
            kv_cache_gb=17.16,
            kv_tokens=524288,
            total_batches=100,
            total_new_tokens=50000,
            total_cached_tokens=10000,
            cache_hit_rate=16.67,
            avg_input_throughput=5000.0,
            max_input_throughput=8000.0,
            max_queue_requests=5,
            max_inflight_requests=10,
        )
        assert node.tp_size == 8
        assert node.kv_cache_gb == 17.16
        assert node.total_new_tokens == 50000
        assert node.cache_hit_rate == 16.67

    def test_decode_node_rollup(self):
        """Test creating a decode NodeRollup with all metrics."""
        node = NodeRollup(
            node_name="node-02",
            worker_type="decode",
            worker_id="w0",
            tp_size=8,
            total_batches=500,
            avg_running_requests=50.0,
            max_running_requests=100,
            avg_gen_throughput=150.0,
            max_gen_throughput=200.0,
            max_queue_requests=10,
            max_transfer_requests=5,
        )
        assert node.worker_type == "decode"
        assert node.avg_gen_throughput == 150.0
        assert node.max_running_requests == 100

    def test_agg_node_rollup(self):
        """Test creating an agg NodeRollup with both prefill and decode metrics."""
        node = NodeRollup(
            node_name="node-03",
            worker_type="agg",
            worker_id="w0",
            tp_size=8,
            total_batches=600,
            total_prefill_batches=100,
            total_decode_batches=500,
            # Prefill stats
            total_new_tokens=50000,
            total_cached_tokens=10000,
            cache_hit_rate=16.67,
            avg_input_throughput=5000.0,
            max_input_throughput=8000.0,
            # Decode stats
            avg_running_requests=50.0,
            max_running_requests=100,
            avg_gen_throughput=150.0,
            max_gen_throughput=200.0,
        )
        assert node.worker_type == "agg"
        assert node.is_agg is True
        assert node.total_prefill_batches == 100
        assert node.total_decode_batches == 500
        assert node.avg_input_throughput == 5000.0
        assert node.avg_gen_throughput == 150.0

    def test_from_node_metrics(self):
        """Test creating NodeRollup from NodeMetrics."""
        from analysis.srtlog.models import BatchMetrics, MemoryMetrics, NodeMetrics

        # Create a mock NodeMetrics with prefill batches
        node_metrics = NodeMetrics(
            node_info={"node": "test-node", "worker_type": "prefill", "worker_id": "w0"},
            batches=[
                BatchMetrics(
                    timestamp="2025-01-22 10:00:00",
                    dp=0,
                    tp=0,
                    ep=0,
                    batch_type="prefill",
                    new_token=1000,
                    cached_token=200,
                    input_throughput=5000.0,
                    queue_req=2,
                    inflight_req=5,
                ),
                BatchMetrics(
                    timestamp="2025-01-22 10:00:01",
                    dp=0,
                    tp=0,
                    ep=0,
                    batch_type="prefill",
                    new_token=1500,
                    cached_token=300,
                    input_throughput=6000.0,
                    queue_req=3,
                    inflight_req=8,
                ),
            ],
            memory_snapshots=[
                MemoryMetrics(
                    timestamp="2025-01-22 10:00:00",
                    dp=0,
                    tp=0,
                    ep=0,
                    metric_type="memory",
                    avail_mem_gb=75.0,
                    mem_usage_gb=107.0,
                    kv_cache_gb=17.16,
                    kv_tokens=524288,
                ),
            ],
            config={"tp_size": 8, "dp_size": 1, "ep_size": 1},
        )

        rollup = NodeRollup.from_node_metrics(node_metrics)

        assert rollup.node_name == "test-node"
        assert rollup.worker_type == "prefill"
        assert rollup.tp_size == 8
        assert rollup.total_batches == 2
        assert rollup.total_new_tokens == 2500  # 1000 + 1500
        assert rollup.total_cached_tokens == 500  # 200 + 300
        assert rollup.avg_input_throughput == 5500.0  # (5000 + 6000) / 2
        assert rollup.max_input_throughput == 6000.0
        assert rollup.max_queue_requests == 3
        assert rollup.max_inflight_requests == 8
        assert rollup.kv_cache_gb == 17.16

        # Check cache hit rate: 500 / (2500 + 500) = 16.67%
        assert rollup.cache_hit_rate == pytest.approx(16.67, rel=0.01)

    def test_from_node_metrics_agg_worker(self):
        """Test creating NodeRollup from agg worker NodeMetrics."""
        from analysis.srtlog.models import BatchMetrics, MemoryMetrics, NodeMetrics

        # Create a mock NodeMetrics with agg worker (has both prefill and decode batches)
        node_metrics = NodeMetrics(
            node_info={"node": "agg-node", "worker_type": "agg", "worker_id": "w0"},
            batches=[
                # Prefill batches
                BatchMetrics(
                    timestamp="2025-01-22 10:00:00",
                    dp=0,
                    tp=0,
                    ep=0,
                    batch_type="prefill",
                    new_token=1000,
                    cached_token=200,
                    input_throughput=5000.0,
                    inflight_req=5,
                ),
                BatchMetrics(
                    timestamp="2025-01-22 10:00:01",
                    dp=0,
                    tp=0,
                    ep=0,
                    batch_type="prefill",
                    new_token=1500,
                    cached_token=300,
                    input_throughput=6000.0,
                    inflight_req=8,
                ),
                # Decode batches
                BatchMetrics(
                    timestamp="2025-01-22 10:00:02",
                    dp=0,
                    tp=0,
                    ep=0,
                    batch_type="decode",
                    running_req=50,
                    gen_throughput=150.0,
                    queue_req=3,
                ),
                BatchMetrics(
                    timestamp="2025-01-22 10:00:03",
                    dp=0,
                    tp=0,
                    ep=0,
                    batch_type="decode",
                    running_req=60,
                    gen_throughput=180.0,
                    queue_req=5,
                ),
            ],
            memory_snapshots=[
                MemoryMetrics(
                    timestamp="2025-01-22 10:00:00",
                    dp=0,
                    tp=0,
                    ep=0,
                    metric_type="memory",
                    kv_cache_gb=20.0,
                ),
            ],
            config={"tp_size": 8},
        )

        rollup = NodeRollup.from_node_metrics(node_metrics)

        assert rollup.node_name == "agg-node"
        assert rollup.worker_type == "agg"
        assert rollup.is_agg is True
        assert rollup.total_batches == 4
        assert rollup.total_prefill_batches == 2
        assert rollup.total_decode_batches == 2

        # Prefill stats
        assert rollup.total_new_tokens == 2500  # 1000 + 1500
        assert rollup.total_cached_tokens == 500  # 200 + 300
        assert rollup.avg_input_throughput == 5500.0  # (5000 + 6000) / 2
        assert rollup.max_input_throughput == 6000.0
        assert rollup.max_inflight_requests == 8

        # Decode stats
        assert rollup.avg_running_requests == 55.0  # (50 + 60) / 2
        assert rollup.max_running_requests == 60
        assert rollup.avg_gen_throughput == 165.0  # (150 + 180) / 2
        assert rollup.max_gen_throughput == 180.0
        assert rollup.max_queue_requests == 5


class TestNodesSummary:
    """Tests for NodesSummary dataclass."""

    def test_empty_summary(self):
        """Test creating an empty NodesSummary."""
        summary = NodesSummary()
        assert summary.total_prefill_nodes == 0
        assert summary.total_decode_nodes == 0
        assert summary.nodes == []

    def test_from_node_metrics_list(self):
        """Test creating NodesSummary from NodeMetrics list."""
        from analysis.srtlog.models import BatchMetrics, NodeMetrics

        nodes = [
            NodeMetrics(
                node_info={"node": "node-01", "worker_type": "prefill", "worker_id": "w0"},
                batches=[
                    BatchMetrics(
                        timestamp="2025-01-22 10:00:00",
                        dp=0,
                        tp=0,
                        ep=0,
                        batch_type="prefill",
                        new_token=1000,
                        cached_token=200,
                        input_throughput=5000.0,
                    ),
                ],
                config={"tp_size": 8},
            ),
            NodeMetrics(
                node_info={"node": "node-02", "worker_type": "decode", "worker_id": "w0"},
                batches=[
                    BatchMetrics(
                        timestamp="2025-01-22 10:00:00",
                        dp=0,
                        tp=0,
                        ep=0,
                        batch_type="decode",
                        running_req=50,
                        gen_throughput=150.0,
                    ),
                ],
                config={"tp_size": 8},
            ),
            NodeMetrics(
                node_info={"node": "node-03", "worker_type": "decode", "worker_id": "w0"},
                batches=[
                    BatchMetrics(
                        timestamp="2025-01-22 10:00:00",
                        dp=0,
                        tp=0,
                        ep=0,
                        batch_type="decode",
                        running_req=60,
                        gen_throughput=180.0,
                    ),
                ],
                config={"tp_size": 8},
            ),
        ]

        summary = NodesSummary.from_node_metrics_list(nodes)

        assert summary.total_prefill_nodes == 1
        assert summary.total_decode_nodes == 2
        assert len(summary.nodes) == 3
        assert summary.total_prefill_tokens == 1000
        assert summary.total_cached_tokens == 200
        assert summary.avg_prefill_input_throughput == 5000.0
        assert summary.avg_decode_gen_throughput == 165.0  # (150 + 180) / 2
        assert summary.max_decode_gen_throughput == 180.0

    def test_from_node_metrics_list_with_agg(self):
        """Test creating NodesSummary from NodeMetrics list including agg workers."""
        from analysis.srtlog.models import BatchMetrics, NodeMetrics

        nodes = [
            # One agg worker
            NodeMetrics(
                node_info={"node": "agg-node-01", "worker_type": "agg", "worker_id": "w0"},
                batches=[
                    BatchMetrics(
                        timestamp="2025-01-22 10:00:00",
                        dp=0,
                        tp=0,
                        ep=0,
                        batch_type="prefill",
                        new_token=1000,
                        cached_token=200,
                        input_throughput=5000.0,
                    ),
                    BatchMetrics(
                        timestamp="2025-01-22 10:00:01",
                        dp=0,
                        tp=0,
                        ep=0,
                        batch_type="decode",
                        running_req=50,
                        gen_throughput=150.0,
                    ),
                ],
                config={"tp_size": 8},
            ),
            # Another agg worker
            NodeMetrics(
                node_info={"node": "agg-node-02", "worker_type": "agg", "worker_id": "w0"},
                batches=[
                    BatchMetrics(
                        timestamp="2025-01-22 10:00:00",
                        dp=0,
                        tp=0,
                        ep=0,
                        batch_type="prefill",
                        new_token=1500,
                        cached_token=300,
                        input_throughput=6000.0,
                    ),
                    BatchMetrics(
                        timestamp="2025-01-22 10:00:01",
                        dp=0,
                        tp=0,
                        ep=0,
                        batch_type="decode",
                        running_req=60,
                        gen_throughput=180.0,
                    ),
                ],
                config={"tp_size": 8},
            ),
        ]

        summary = NodesSummary.from_node_metrics_list(nodes)

        # Check counts
        assert summary.total_prefill_nodes == 0
        assert summary.total_decode_nodes == 0
        assert summary.total_agg_nodes == 2
        assert len(summary.nodes) == 2

        # Aggregated stats should include agg nodes
        assert summary.total_prefill_tokens == 2500  # 1000 + 1500
        assert summary.total_cached_tokens == 500  # 200 + 300
        assert summary.avg_prefill_input_throughput == 5500.0  # (5000 + 6000) / 2
        assert summary.max_prefill_input_throughput == 6000.0
        assert summary.avg_decode_gen_throughput == 165.0  # (150 + 180) / 2
        assert summary.max_decode_gen_throughput == 180.0


class TestRollupResult:
    """Tests for RollupResult dataclass."""

    def test_minimal_result(self):
        """Test creating a result with minimal required fields."""
        result = RollupResult(concurrency=100, output_tps=5000.0)
        assert result.concurrency == 100
        assert result.output_tps == 5000.0
        assert result.mean_ttft_ms is None
        assert result.total_tps is None

    def test_full_result(self):
        """Test creating a result with all fields populated."""
        result = RollupResult(
            concurrency=100,
            output_tps=5000.0,
            total_tps=6000.0,
            request_throughput=50.0,
            mean_ttft_ms=150.0,
            mean_tpot_ms=20.0,
            mean_itl_ms=18.0,
            p99_ttft_ms=300.0,
            p99_itl_ms=25.0,
            total_input_tokens=100000,
            total_output_tokens=200000,
            duration=60.0,
            completed=1000,
            num_prompts=1000,
        )
        assert result.concurrency == 100
        assert result.output_tps == 5000.0
        assert result.total_tps == 6000.0
        assert result.mean_ttft_ms == 150.0
        assert result.p99_ttft_ms == 300.0


class TestRollupSummary:
    """Tests for RollupSummary dataclass."""

    def test_compute_summary_stats_empty(self):
        """Test summary stats with no results."""
        summary = RollupSummary(
            job_id="12345",
            job_name="test-job",
            generated_at="2025-01-22 10:00:00",
            model_path="/models/test",
            model_name="test-model",
            precision="fp8",
            gpu_type="B200",
            gpus_per_node=8,
            backend_type="sglang",
            frontend_type="sglang",
            is_disaggregated=True,
            total_nodes=4,
            total_gpus=32,
            benchmark_type="sa-bench",
            isl=1024,
            osl=1024,
        )
        summary.compute_summary_stats()
        assert summary.max_output_tps is None
        assert summary.min_mean_ttft_ms is None

    def test_compute_summary_stats_with_results(self):
        """Test summary stats computation from results."""
        summary = RollupSummary(
            job_id="12345",
            job_name="test-job",
            generated_at="2025-01-22 10:00:00",
            model_path="/models/test",
            model_name="test-model",
            precision="fp8",
            gpu_type="B200",
            gpus_per_node=8,
            backend_type="sglang",
            frontend_type="sglang",
            is_disaggregated=True,
            total_nodes=4,
            total_gpus=32,
            benchmark_type="sa-bench",
            isl=1024,
            osl=1024,
            results=[
                RollupResult(concurrency=50, output_tps=3000.0, mean_ttft_ms=100.0, mean_itl_ms=20.0),
                RollupResult(concurrency=100, output_tps=5000.0, mean_ttft_ms=150.0, mean_itl_ms=25.0),
                RollupResult(concurrency=200, output_tps=4500.0, mean_ttft_ms=250.0, mean_itl_ms=30.0),
            ],
        )
        summary.compute_summary_stats()

        assert summary.max_output_tps == 5000.0
        assert summary.min_mean_ttft_ms == 100.0
        assert summary.min_mean_itl_ms == 20.0


class TestRollupStageMixin:
    """Tests for RollupStageMixin functionality."""

    def test_collect_benchmark_results(self, tmp_path):
        """Test collecting benchmark results from directories."""
        # Create mock benchmark result directories
        bench_dir = tmp_path / "sa-bench_isl_1024_osl_1024"
        bench_dir.mkdir()

        # Create mock result JSONs
        for concurrency in [50, 100, 200]:
            result_file = bench_dir / f"result_c{concurrency}.json"
            result_file.write_text(
                json.dumps(
                    {
                        "max_concurrency": concurrency,
                        "output_throughput": 1000.0 * concurrency / 50,
                        "total_token_throughput": 1200.0 * concurrency / 50,
                        "mean_ttft_ms": 100.0 + concurrency,
                        "mean_itl_ms": 15.0 + concurrency / 10,
                        "request_rate": f"c{concurrency}",
                    }
                )
            )

        # Create a mock mixin instance
        class MockBenchmarkConfig:
            type = "sa-bench"

        class MockConfig:
            benchmark = MockBenchmarkConfig()

        class MockOrchestrator(RollupStageMixin):
            def __init__(self, log_dir):
                self._log_dir = log_dir

            @property
            def config(self):
                return MockConfig()

            @property
            def runtime(self):
                class MockRuntime:
                    log_dir = self._log_dir

                return MockRuntime()

            @property
            def endpoints(self):
                return []

        orchestrator = MockOrchestrator(tmp_path)
        results = orchestrator._collect_benchmark_results()

        assert len(results) == 3
        # Results should be sorted by concurrency
        assert results[0]["max_concurrency"] == 50
        assert results[1]["max_concurrency"] == 100
        assert results[2]["max_concurrency"] == 200

    def test_collect_benchmark_results_empty(self, tmp_path):
        """Test collecting when no benchmark results exist."""

        class MockBenchmarkConfig:
            type = "sa-bench"

        class MockConfig:
            benchmark = MockBenchmarkConfig()

        class MockOrchestrator(RollupStageMixin):
            def __init__(self, log_dir):
                self._log_dir = log_dir

            @property
            def config(self):
                return MockConfig()

            @property
            def runtime(self):
                class MockRuntime:
                    log_dir = self._log_dir

                return MockRuntime()

            @property
            def endpoints(self):
                return []

        orchestrator = MockOrchestrator(tmp_path)
        results = orchestrator._collect_benchmark_results()

        assert len(results) == 0

    def test_write_rollup(self, tmp_path):
        """Test writing rollup summary to JSON."""
        summary = RollupSummary(
            job_id="12345",
            job_name="test-job",
            generated_at="2025-01-22 10:00:00",
            model_path="/models/test",
            model_name="test-model",
            precision="fp8",
            gpu_type="B200",
            gpus_per_node=8,
            backend_type="sglang",
            frontend_type="sglang",
            is_disaggregated=True,
            total_nodes=4,
            total_gpus=32,
            benchmark_type="sa-bench",
            isl=1024,
            osl=1024,
            prefill_nodes=1,
            decode_nodes=3,
            prefill_workers=1,
            decode_workers=3,
            prefill_gpus=8,
            decode_gpus=24,
            results=[
                RollupResult(concurrency=100, output_tps=5000.0, mean_ttft_ms=150.0),
            ],
            tags=["test", "example"],
        )
        summary.compute_summary_stats()

        class MockOrchestrator(RollupStageMixin):
            @property
            def runtime(self):
                return None

            @property
            def endpoints(self):
                return []

        orchestrator = MockOrchestrator()
        rollup_path = tmp_path / "rollup.json"
        orchestrator._write_rollup(summary, rollup_path)

        # Verify the file was written
        assert rollup_path.exists()

        # Verify the content
        with open(rollup_path) as f:
            data = json.load(f)

        assert data["job_id"] == "12345"
        assert data["job_name"] == "test-job"
        assert data["model_name"] == "test-model"
        assert data["is_disaggregated"] is True
        assert data["total_gpus"] == 32
        assert data["prefill_nodes"] == 1
        assert data["decode_nodes"] == 3
        assert len(data["results"]) == 1
        assert data["results"][0]["concurrency"] == 100
        assert data["results"][0]["output_tps"] == 5000.0
        assert data["max_output_tps"] == 5000.0
        assert data["tags"] == ["test", "example"]


class TestRollupIntegration:
    """Integration tests for rollup with full mock config."""

    def test_full_rollup_workflow(self, tmp_path):
        """Test the complete rollup workflow with mocked config."""
        from dataclasses import dataclass, field

        # Create mock benchmark results
        bench_dir = tmp_path / "sa-bench_isl_1024_osl_1024"
        bench_dir.mkdir()

        for concurrency in [50, 100, 200]:
            result_file = bench_dir / f"result_c{concurrency}.json"
            result_file.write_text(
                json.dumps(
                    {
                        "max_concurrency": concurrency,
                        "output_throughput": 1000.0 * concurrency / 50,
                        "total_token_throughput": 1200.0 * concurrency / 50,
                        "mean_ttft_ms": 100.0 + concurrency,
                        "mean_itl_ms": 15.0 + concurrency / 10,
                        "p99_ttft_ms": 200.0 + concurrency * 2,
                        "p99_itl_ms": 30.0 + concurrency / 5,
                        "duration": 60.0,
                        "completed": concurrency * 10,
                        "num_prompts": concurrency * 10,
                    }
                )
            )

        # Create mock orchestrator with full config
        @dataclass
        class MockResourceConfig:
            is_disaggregated: bool = True
            prefill_gpus: int = 8
            decode_gpus: int = 24
            agg_nodes: int | None = None
            gpus_per_node: int = 8
            gpu_type: str = "B200"
            total_nodes: int = 4
            prefill_nodes: int = 1
            decode_nodes: int = 3
            num_prefill: int = 1
            num_decode: int = 3
            num_agg: int | None = None

        @dataclass
        class MockBenchmarkConfig:
            type: str = "sa-bench"
            isl: int = 1024
            osl: int = 1024
            concurrencies: str = "50x100x200"

            def get_concurrency_list(self):
                return [int(c) for c in self.concurrencies.split("x")]

        @dataclass
        class MockModelConfig:
            precision: str = "fp8"

        @dataclass
        class MockFrontendConfig:
            type: str = "sglang"

        @dataclass
        class MockConfig:
            name: str = "test-job"
            served_model_name: str = "deepseek-v3"
            backend_type: str = "sglang"
            resources: MockResourceConfig = field(default_factory=MockResourceConfig)
            benchmark: MockBenchmarkConfig = field(default_factory=MockBenchmarkConfig)
            model: MockModelConfig = field(default_factory=MockModelConfig)
            frontend: MockFrontendConfig = field(default_factory=MockFrontendConfig)

        @dataclass
        class MockRuntime:
            job_id: str = "12345"
            log_dir: Path = field(default_factory=Path)
            model_path: Path = field(default_factory=lambda: Path("/models/deepseek-v3"))

        class MockOrchestrator(RollupStageMixin):
            def __init__(self, config, runtime):
                self._config = config
                self._runtime = runtime

            @property
            def config(self):
                return self._config

            @property
            def runtime(self):
                return self._runtime

            @property
            def endpoints(self):
                return []

        config = MockConfig()
        runtime = MockRuntime(log_dir=tmp_path)
        orchestrator = MockOrchestrator(config, runtime)

        # Run rollup
        rollup_path = orchestrator.run_rollup(tags=["integration-test"])

        # Verify
        assert rollup_path is not None
        assert rollup_path.exists()

        with open(rollup_path) as f:
            data = json.load(f)

        # Verify summary
        assert data["job_id"] == "12345"
        assert data["job_name"] == "test-job"
        assert data["model_name"] == "deepseek-v3"
        assert data["is_disaggregated"] is True
        assert data["total_gpus"] == 32  # 8 + 24
        assert data["benchmark_type"] == "sa-bench"
        assert data["isl"] == 1024
        assert data["osl"] == 1024
        assert data["concurrencies"] == [50, 100, 200]

        # Verify results
        assert len(data["results"]) == 3
        assert data["max_output_tps"] == 4000.0  # 1000 * 200/50

        # Verify tags
        assert data["tags"] == ["integration-test"]

    def test_rollup_with_node_logs(self, tmp_path):
        """Test rollup with actual node log files parsed by NodeAnalyzer."""
        from dataclasses import dataclass, field

        # Create mock benchmark results
        bench_dir = tmp_path / "sa-bench_isl_1024_osl_1024"
        bench_dir.mkdir()

        result_file = bench_dir / "result_c100.json"
        result_file.write_text(
            json.dumps(
                {
                    "max_concurrency": 100,
                    "output_throughput": 5000.0,
                    "mean_ttft_ms": 150.0,
                    "mean_itl_ms": 20.0,
                }
            )
        )

        # Create mock prefill log file (matches NodeAnalyzer expected format)
        prefill_log = tmp_path / "node-01_prefill_w0.err"
        prefill_log.write_text(
            """[2025-01-22 10:00:00 DP0 TP0 EP0] Prefill batch, #new-seq: 10, #new-token: 1024, #cached-token: 256, token usage: 0.50, #running-req: 5, #queue-req: 2, #prealloc-req: 0, #inflight-req: 3, input throughput (token/s): 5000.00,
[2025-01-22 10:00:01 DP0 TP0 EP0] Load weight end. type=DeepseekV3ForCausalLM, dtype=torch.bfloat16, avail mem=75.11 GB, mem usage=107.07 GB.
[2025-01-22 10:00:02 DP0 TP0 EP0] KV Cache is allocated. #tokens: 524288, KV size: 17.16 GB
"""
        )

        # Create mock decode log file
        decode_log = tmp_path / "node-02_decode_w0.err"
        decode_log.write_text(
            """[2025-01-22 10:00:00 DP31 TP31 EP31] Decode batch, #running-req: 50, #token: 5000, token usage: 0.50, pre-allocated usage: 0.10, #prealloc-req: 2, #transfer-req: 1, #queue-req: 3, gen throughput (token/s): 150.00,
"""
        )

        # Mock config classes
        @dataclass
        class MockResourceConfig:
            is_disaggregated: bool = True
            prefill_gpus: int = 8
            decode_gpus: int = 8
            agg_nodes: int | None = None
            gpus_per_node: int = 8
            gpu_type: str = "B200"
            total_nodes: int = 2
            prefill_nodes: int = 1
            decode_nodes: int = 1
            num_prefill: int = 1
            num_decode: int = 1
            num_agg: int | None = None

        @dataclass
        class MockBenchmarkConfig:
            type: str = "sa-bench"
            isl: int = 1024
            osl: int = 1024
            concurrencies: str = "100"

            def get_concurrency_list(self):
                return [100]

        @dataclass
        class MockModelConfig:
            precision: str = "fp8"

        @dataclass
        class MockFrontendConfig:
            type: str = "sglang"

        @dataclass
        class MockConfig:
            name: str = "test-job"
            served_model_name: str = "deepseek-v3"
            backend_type: str = "sglang"
            resources: MockResourceConfig = field(default_factory=MockResourceConfig)
            benchmark: MockBenchmarkConfig = field(default_factory=MockBenchmarkConfig)
            model: MockModelConfig = field(default_factory=MockModelConfig)
            frontend: MockFrontendConfig = field(default_factory=MockFrontendConfig)

        @dataclass
        class MockRuntime:
            job_id: str = "12345"
            log_dir: Path = field(default_factory=Path)
            model_path: Path = field(default_factory=lambda: Path("/models/deepseek-v3"))

        class MockOrchestrator(RollupStageMixin):
            def __init__(self, config, runtime):
                self._config = config
                self._runtime = runtime

            @property
            def config(self):
                return self._config

            @property
            def runtime(self):
                return self._runtime

            @property
            def endpoints(self):
                return []

        config = MockConfig()
        runtime = MockRuntime(log_dir=tmp_path)
        orchestrator = MockOrchestrator(config, runtime)

        # Run rollup
        rollup_path = orchestrator.run_rollup(tags=["node-test"])

        assert rollup_path is not None
        assert rollup_path.exists()

        with open(rollup_path) as f:
            data = json.load(f)

        # Verify node summary is present
        assert data["nodes_summary"] is not None
        nodes_summary = data["nodes_summary"]

        assert nodes_summary["total_prefill_nodes"] == 1
        assert nodes_summary["total_decode_nodes"] == 1
        assert len(nodes_summary["nodes"]) == 2

        # Find prefill and decode nodes
        prefill_node = next((n for n in nodes_summary["nodes"] if n["worker_type"] == "prefill"), None)
        decode_node = next((n for n in nodes_summary["nodes"] if n["worker_type"] == "decode"), None)

        assert prefill_node is not None
        assert prefill_node["node_name"] == "node-01"
        assert prefill_node["total_new_tokens"] == 1024
        assert prefill_node["total_cached_tokens"] == 256
        assert prefill_node["kv_cache_gb"] == 17.16

        assert decode_node is not None
        assert decode_node["node_name"] == "node-02"
        assert decode_node["max_running_requests"] == 50
        assert decode_node["avg_gen_throughput"] == 150.0


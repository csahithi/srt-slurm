# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for metrics collection stage logic."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.cli.mixins.frontend_stage import FrontendTopology
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import FrontendConfig, MetricsConfig, ResourceConfig, SrtConfig


def make_config(
    *,
    metrics_enabled: bool = False,
    prometheus_port: int = 9090,
    scrape_interval: str = "5s",
) -> SrtConfig:
    """Create a minimal SrtConfig for testing."""
    return SrtConfig(
        name="test-config",
        model={"path": "test-model", "container": "test.sqsh", "precision": "fp16"},
        resources=ResourceConfig(
            gpu_type="a100",
            gpus_per_node=8,
            prefill_nodes=1,
            decode_nodes=1,
            prefill_workers=1,
            decode_workers=1,
        ),
        frontend=FrontendConfig(type="dynamo"),
        metrics=MetricsConfig(
            enabled=metrics_enabled,
            prometheus_port=prometheus_port,
            scrape_interval=scrape_interval,
        ),
    )


def make_runtime(nodes: list[str]) -> RuntimeContext:
    """Create a minimal RuntimeContext for testing."""
    return RuntimeContext(
        job_id="12345",
        run_name="test-run",
        nodes=Nodes(head=nodes[0], bench=nodes[0], worker=tuple(nodes)),
        head_node_ip="10.0.0.1",
        log_dir=Path("/tmp/logs"),
        model_path=Path("/models/test-model"),
        container_image=Path("/path/to/container.sqsh"),
        gpus_per_node=8,
        network_interface=None,
        container_mounts={},
        environment={},
    )


def make_frontend_topology(nodes: list[str], frontend_port: int = 8000) -> FrontendTopology:
    """Create a minimal FrontendTopology for testing."""
    return FrontendTopology(
        nginx_node=None,
        frontend_nodes=nodes,
        frontend_port=frontend_port,
        public_port=8000,
    )


class TestMetricsNodeSelection:
    """Tests for metrics node selection logic."""

    def test_single_node_uses_head(self):
        """Single node: Prometheus runs on head (only option)."""
        config = make_config(metrics_enabled=True)
        runtime = make_runtime(["node0"])

        orchestrator = SweepOrchestrator(config=config, runtime=runtime)
        node = orchestrator.select_metrics_node()

        assert node == "node0"

    def test_two_nodes_uses_non_head(self):
        """Two nodes: Prometheus runs on non-head node."""
        config = make_config(metrics_enabled=True)
        runtime = make_runtime(["node0", "node1"])

        orchestrator = SweepOrchestrator(config=config, runtime=runtime)
        node = orchestrator.select_metrics_node()

        assert node == "node1"

    def test_many_nodes_uses_last(self):
        """Many nodes: Prometheus runs on last worker node."""
        config = make_config(metrics_enabled=True)
        runtime = make_runtime(["node0", "node1", "node2", "node3", "node4"])

        orchestrator = SweepOrchestrator(config=config, runtime=runtime)
        node = orchestrator.select_metrics_node()

        # Should use the last node to avoid conflicting with frontends
        assert node == "node4"

    def test_excludes_head_node(self):
        """Metrics node selection excludes head node when possible."""
        config = make_config(metrics_enabled=True)
        runtime = make_runtime(["node0", "node1", "node2"])

        orchestrator = SweepOrchestrator(config=config, runtime=runtime)
        node = orchestrator.select_metrics_node()

        assert node != "node0"


class TestPrometheusConfigGeneration:
    """Tests for Prometheus config generation."""

    def test_generates_valid_yaml(self):
        """Generated config is valid YAML."""
        config = make_config(metrics_enabled=True)
        runtime = make_runtime(["node0", "node1"])
        topology = make_frontend_topology(["node0"])

        orchestrator = SweepOrchestrator(config=config, runtime=runtime)
        prometheus_config = orchestrator.generate_prometheus_config(topology)

        # Should be valid YAML
        parsed = yaml.safe_load(prometheus_config)
        assert parsed is not None
        assert "global" in parsed
        assert "scrape_configs" in parsed

    def test_includes_scrape_interval(self):
        """Config includes configured scrape interval."""
        config = make_config(metrics_enabled=True, scrape_interval="10s")
        runtime = make_runtime(["node0", "node1"])
        topology = make_frontend_topology(["node0"])

        orchestrator = SweepOrchestrator(config=config, runtime=runtime)
        prometheus_config = orchestrator.generate_prometheus_config(topology)

        parsed = yaml.safe_load(prometheus_config)
        assert parsed["global"]["scrape_interval"] == "10s"

    def test_includes_worker_targets(self):
        """Config includes targets for all worker processes."""
        config = make_config(metrics_enabled=True)
        runtime = make_runtime(["node0", "node1"])
        topology = make_frontend_topology(["node0"])

        orchestrator = SweepOrchestrator(config=config, runtime=runtime)
        prometheus_config = orchestrator.generate_prometheus_config(topology)

        parsed = yaml.safe_load(prometheus_config)
        scrape_configs = parsed["scrape_configs"]

        # Find the workers job
        worker_job = next((j for j in scrape_configs if j["job_name"] == "dynamo_workers"), None)
        assert worker_job is not None

        # Should have targets for the backend processes
        static_configs = worker_job["static_configs"]
        assert len(static_configs) > 0

        # Each target should have proper labels
        for target_config in static_configs:
            labels = target_config.get("labels", {})
            assert "worker_mode" in labels
            assert "worker_index" in labels
            assert "node" in labels

    def test_includes_frontend_targets(self):
        """Config includes targets for frontend processes."""
        config = make_config(metrics_enabled=True)
        runtime = make_runtime(["node0", "node1"])
        topology = make_frontend_topology(["node0", "node1"], frontend_port=8080)

        orchestrator = SweepOrchestrator(config=config, runtime=runtime)
        prometheus_config = orchestrator.generate_prometheus_config(topology)

        parsed = yaml.safe_load(prometheus_config)
        scrape_configs = parsed["scrape_configs"]

        # Find the frontends job
        frontend_job = next((j for j in scrape_configs if j["job_name"] == "frontends"), None)
        assert frontend_job is not None

        # Should have targets for frontends
        static_configs = frontend_job["static_configs"]
        assert len(static_configs) == 2  # Two frontends

        # Each frontend target should have proper labels
        for target_config in static_configs:
            labels = target_config.get("labels", {})
            assert "frontend_index" in labels
            assert "node" in labels

    def test_uses_sys_port_for_worker_targets(self):
        """Worker targets use sys_port (DYN_SYSTEM_PORT) for metrics endpoint."""
        config = make_config(metrics_enabled=True)
        runtime = make_runtime(["node0", "node1"])
        topology = make_frontend_topology(["node0"])

        orchestrator = SweepOrchestrator(config=config, runtime=runtime)
        prometheus_config = orchestrator.generate_prometheus_config(topology)

        parsed = yaml.safe_load(prometheus_config)
        worker_job = next((j for j in parsed["scrape_configs"] if j["job_name"] == "dynamo_workers"), None)
        static_configs = worker_job["static_configs"]

        # All targets should have port format node:port
        for target_config in static_configs:
            targets = target_config["targets"]
            for target in targets:
                assert ":" in target  # Should be host:port format

    def test_uses_frontend_port_for_frontend_targets(self):
        """Frontend targets use frontend_port for metrics endpoint."""
        config = make_config(metrics_enabled=True)
        runtime = make_runtime(["node0", "node1"])
        topology = make_frontend_topology(["node1"], frontend_port=8080)

        orchestrator = SweepOrchestrator(config=config, runtime=runtime)
        prometheus_config = orchestrator.generate_prometheus_config(topology)

        parsed = yaml.safe_load(prometheus_config)
        frontend_job = next((j for j in parsed["scrape_configs"] if j["job_name"] == "frontends"), None)
        static_configs = frontend_job["static_configs"]

        # All frontend targets should use the frontend port
        for target_config in static_configs:
            targets = target_config["targets"]
            for target in targets:
                assert target.endswith(":8080")


class TestMetricsCollectionStartup:
    """Tests for starting metrics collection."""

    def test_disabled_returns_empty(self):
        """When metrics disabled, start_metrics_collection returns empty dict."""
        config = make_config(metrics_enabled=False)
        runtime = make_runtime(["node0", "node1"])
        topology = make_frontend_topology(["node0"])

        orchestrator = SweepOrchestrator(config=config, runtime=runtime)
        processes = orchestrator.start_metrics_collection(topology)

        assert processes == {}

    @patch("srtctl.cli.mixins.metrics_stage.start_srun_process")
    def test_enabled_starts_prometheus(self, mock_srun, tmp_path):
        """When metrics enabled, starts Prometheus on selected node."""
        mock_srun.return_value = MagicMock()

        config = make_config(metrics_enabled=True)
        runtime = RuntimeContext(
            job_id="12345",
            run_name="test-run",
            nodes=Nodes(head="node0", bench="node0", worker=("node0", "node1")),
            head_node_ip="10.0.0.1",
            log_dir=tmp_path,
            model_path=Path("/models/test-model"),
            container_image=Path("/path/to/container.sqsh"),
            gpus_per_node=8,
            network_interface=None,
            container_mounts={},
            environment={},
        )
        topology = make_frontend_topology(["node0"])

        orchestrator = SweepOrchestrator(config=config, runtime=runtime)
        processes = orchestrator.start_metrics_collection(topology)

        assert "prometheus" in processes
        assert processes["prometheus"].node == "node1"  # Non-head node
        assert processes["prometheus"].critical is False  # Metrics are non-critical

        # Verify config file was written
        assert (tmp_path / "prometheus.yml").exists()

    @patch("srtctl.cli.mixins.metrics_stage.start_srun_process")
    def test_uses_configured_port(self, mock_srun, tmp_path):
        """Prometheus uses configured port."""
        mock_srun.return_value = MagicMock()

        config = make_config(metrics_enabled=True, prometheus_port=9999)
        runtime = RuntimeContext(
            job_id="12345",
            run_name="test-run",
            nodes=Nodes(head="node0", bench="node0", worker=("node0", "node1")),
            head_node_ip="10.0.0.1",
            log_dir=tmp_path,
            model_path=Path("/models/test-model"),
            container_image=Path("/path/to/container.sqsh"),
            gpus_per_node=8,
            network_interface=None,
            container_mounts={},
            environment={},
        )
        topology = make_frontend_topology(["node0"])

        orchestrator = SweepOrchestrator(config=config, runtime=runtime)
        orchestrator.start_metrics_collection(topology)

        # Check srun was called with correct port in command
        call_args = mock_srun.call_args
        command = call_args.kwargs["command"]
        assert any("9999" in str(arg) for arg in command)

    @patch("srtctl.cli.mixins.metrics_stage.start_srun_process")
    def test_prometheus_config_includes_frontends(self, mock_srun, tmp_path):
        """Generated Prometheus config includes frontend targets."""
        mock_srun.return_value = MagicMock()

        config = make_config(metrics_enabled=True)
        runtime = RuntimeContext(
            job_id="12345",
            run_name="test-run",
            nodes=Nodes(head="node0", bench="node0", worker=("node0", "node1")),
            head_node_ip="10.0.0.1",
            log_dir=tmp_path,
            model_path=Path("/models/test-model"),
            container_image=Path("/path/to/container.sqsh"),
            gpus_per_node=8,
            network_interface=None,
            container_mounts={},
            environment={},
        )
        topology = make_frontend_topology(["node0", "node1"], frontend_port=8080)

        orchestrator = SweepOrchestrator(config=config, runtime=runtime)
        orchestrator.start_metrics_collection(topology)

        # Read the generated config
        config_content = (tmp_path / "prometheus.yml").read_text()
        parsed = yaml.safe_load(config_content)

        # Should have both worker and frontend jobs
        job_names = [j["job_name"] for j in parsed["scrape_configs"]]
        assert "dynamo_workers" in job_names
        assert "frontends" in job_names

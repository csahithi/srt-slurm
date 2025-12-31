# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Metrics stage mixin for SweepOrchestrator.

Handles Prometheus metrics collection from dynamo worker and frontend endpoints.
Following the ignition tachometer pattern.
"""

import logging
from typing import TYPE_CHECKING

import yaml

from srtctl.core.processes import ManagedProcess, NamedProcesses
from srtctl.core.slurm import start_srun_process

if TYPE_CHECKING:
    from srtctl.cli.mixins.frontend_stage import FrontendTopology
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig
    from srtctl.core.topology import Endpoint, Process

logger = logging.getLogger(__name__)


class MetricsStageMixin:
    """Mixin for metrics collection stage.

    Launches a Prometheus server to scrape metrics from all dynamo worker
    and frontend endpoints. The Prometheus server runs on a non-head node
    when possible.

    Requires:
        self.config: SrtConfig
        self.runtime: RuntimeContext
        self.backend_processes: list[Process]
        self.endpoints: list[Endpoint]
    """

    # Type hints for mixin dependencies
    config: "SrtConfig"
    runtime: "RuntimeContext"

    @property
    def backend_processes(self) -> list["Process"]:
        """Compute physical process topology from endpoints (cached)."""
        ...

    @property
    def endpoints(self) -> list["Endpoint"]:
        """Endpoint allocation topology."""
        ...

    def select_metrics_node(self) -> str:
        """Select a node for Prometheus metrics collection.

        Node selection priority:
        1. Exclude head node (runs NATS/etcd)
        2. Prefer later worker nodes (less likely to be frontends)
        3. Fallback to head node if no other option

        Returns:
            Hostname of the selected node for Prometheus.
        """
        head = self.runtime.nodes.head
        workers = self.runtime.nodes.worker

        # Prefer worker nodes that aren't the head node
        candidates = [n for n in workers if n != head]

        # TODO: ishan
        # If we have multiple candidates, prefer later nodes (less likely to be frontend)
        if len(candidates) > 1:
            return candidates[-1]  # Last worker node
        elif candidates:
            return candidates[0]
        else:
            # Only head available - use it as fallback
            logger.warning("No non-head nodes available for Prometheus, using head node")
            return head

    def generate_prometheus_config(self, frontend_topology: "FrontendTopology") -> str:
        """Generate Prometheus configuration YAML.

        Creates scrape targets for:
        - All worker endpoints via sys_port (DYN_SYSTEM_PORT)
        - All frontend endpoints via frontend_port

        Both expose /metrics endpoint.

        Args:
            frontend_topology: Frontend topology with node/port info.

        Returns:
            Prometheus configuration as YAML string.
        """
        metrics_config = self.config.metrics

        # Build scrape targets from backend processes
        worker_targets: list[dict] = []
        for process in self.backend_processes:
            target = {
                "targets": [f"{process.node}:{process.sys_port}"],
                "labels": {
                    "worker_mode": process.endpoint_mode,
                    "worker_index": str(process.endpoint_index),
                    "node": process.node,
                    "node_rank": str(process.node_rank),
                },
            }
            worker_targets.append(target)

        # Build scrape targets from frontends
        frontend_targets: list[dict] = []
        for idx, node in enumerate(frontend_topology.frontend_nodes):
            target = {
                "targets": [f"{node}:{frontend_topology.frontend_port}"],
                "labels": {
                    "frontend_index": str(idx),
                    "node": node,
                },
            }
            frontend_targets.append(target)

        # Build the Prometheus config with separate jobs for workers and frontends
        scrape_configs = [
            {
                "job_name": "dynamo_workers",
                "static_configs": worker_targets,
                "metrics_path": "/metrics",
            },
        ]

        # Only add frontend job if we have frontends
        if frontend_targets:
            scrape_configs.append(
                {
                    "job_name": "frontends",
                    "static_configs": frontend_targets,
                    "metrics_path": "/metrics",
                }
            )

        prometheus_config = {
            "global": {
                "scrape_interval": metrics_config.scrape_interval,
                "evaluation_interval": metrics_config.scrape_interval,
            },
            "scrape_configs": scrape_configs,
        }

        return yaml.dump(prometheus_config, default_flow_style=False)

    def start_metrics_collection(self, frontend_topology: "FrontendTopology") -> NamedProcesses:
        """Start Prometheus metrics collection.

        Launches Prometheus on a non-head node to scrape metrics from all
        dynamo worker and frontend endpoints.

        Args:
            frontend_topology: Frontend topology for scraping frontend metrics.

        Returns:
            Dictionary of process names to ManagedProcess objects.
        """
        processes: NamedProcesses = {}

        # Check if metrics collection is enabled
        if not self.config.metrics.enabled:
            logger.info("Metrics collection not enabled")
            return processes

        logger.info("Starting Prometheus metrics collection")

        # Select node for Prometheus
        prometheus_node = self.select_metrics_node()
        logger.info("Prometheus node: %s", prometheus_node)

        # Generate Prometheus config
        prometheus_config_content = self.generate_prometheus_config(frontend_topology)
        prometheus_config_path = self.runtime.log_dir / "prometheus.yml"
        prometheus_config_path.write_text(prometheus_config_content)
        logger.info("Generated Prometheus config: %s", prometheus_config_path)

        # Log file for Prometheus
        prometheus_log = self.runtime.log_dir / f"prometheus_{prometheus_node}.out"

        # Prometheus command
        # Using the container path for config since log_dir is mounted at /logs
        prometheus_port = self.config.metrics.prometheus_port
        cmd = [
            "prometheus",
            "--config.file=/logs/prometheus.yml",
            f"--web.listen-address=:{prometheus_port}",
            "--storage.tsdb.path=/logs/prometheus_data",
            "--storage.tsdb.retention.time=1h",
        ]

        logger.info("Prometheus port: %d", prometheus_port)
        logger.info("Prometheus log: %s", prometheus_log)

        proc = start_srun_process(
            command=cmd,
            nodelist=[prometheus_node],
            output=str(prometheus_log),
            container_image=str(self.runtime.container_image),
            container_mounts=self.runtime.container_mounts,
        )

        processes["prometheus"] = ManagedProcess(
            name="prometheus",
            popen=proc,
            log_file=prometheus_log,
            node=prometheus_node,
            critical=False,  # Metrics collection is non-critical
        )

        logger.info("Prometheus started on %s:%d", prometheus_node, prometheus_port)
        return processes

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Sidecar stage mixin for SweepOrchestrator.

Handles starting auxiliary sidecar processes (e.g., custom Dynamo router, processor).
Sidecars run on the head node after infrastructure (ETCD/NATS) and before the frontend.
"""

import logging
import shlex
from typing import TYPE_CHECKING

from srtctl.core.processes import ManagedProcess
from srtctl.core.slurm import start_srun_process

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SidecarConfig, SrtConfig

logger = logging.getLogger(__name__)


class SidecarStageMixin:
    """Mixin for sidecar process startup stage.

    Sidecars are auxiliary long-running processes that run on the head node.
    They start after infrastructure (ETCD/NATS) and before the frontend.

    Use cases:
        - Custom Dynamo router (Thompson sampling)
        - Custom Dynamo processor
        - Metrics collectors

    Requires:
        self.config: SrtConfig
        self.runtime: RuntimeContext
    """

    # Type hints for mixin dependencies
    config: "SrtConfig"
    runtime: "RuntimeContext"

    def start_sidecars(self) -> list[ManagedProcess]:
        """Start all configured sidecar processes.

        Returns:
            List of ManagedProcess instances for all sidecars.
        """
        if not self.config.sidecars:
            logger.debug("No sidecars configured")
            return []

        logger.info("Starting %d sidecar(s)", len(self.config.sidecars))
        processes: list[ManagedProcess] = []

        for name, sidecar_config in self.config.sidecars.items():
            proc = self._start_sidecar(name, sidecar_config)
            processes.append(proc)

        return processes

    def _start_sidecar(self, name: str, config: "SidecarConfig") -> ManagedProcess:
        """Start a single sidecar process.

        Args:
            name: Sidecar name (used for logging and process identification)
            config: Sidecar configuration

        Returns:
            ManagedProcess for the sidecar
        """
        head_node = self.runtime.nodes.head
        logger.info("Starting sidecar '%s' on %s", name, head_node)

        sidecar_log = self.runtime.log_dir / f"sidecar_{name}.out"

        # Determine container image (sidecar-specific > model.container)
        container_image = config.container or str(self.runtime.container_image)

        # Parse command string into list
        cmd = shlex.split(config.command)

        # Build environment variables
        env_to_set = {
            "ETCD_ENDPOINTS": f"http://{self.runtime.nodes.infra}:2379",
            "NATS_SERVER": f"nats://{self.runtime.nodes.infra}:4222",
        }
        env_to_set.update(config.env)

        logger.info("Sidecar '%s' command: %s", name, config.command)
        logger.info("Sidecar '%s' container: %s", name, container_image)
        logger.info("Sidecar '%s' log: %s", name, sidecar_log)

        proc = start_srun_process(
            command=cmd,
            nodelist=[head_node],
            output=str(sidecar_log),
            container_image=container_image,
            container_mounts=self.runtime.container_mounts,
            env_to_set=env_to_set,
        )

        return ManagedProcess(
            name=f"sidecar_{name}",
            popen=proc,
            log_file=sidecar_log,
            node=head_node,
            critical=True,
        )

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Main orchestration script for benchmark sweeps.

This script is called from within the sbatch job and coordinates:
1. Starting head node infrastructure (NATS, etcd)
2. Starting backend workers (prefill/decode/agg)
3. Starting frontends and nginx
4. Running benchmarks
5. Cleanup
"""

import argparse
import functools
import shlex
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from srtctl.core.config import load_config
from srtctl.core.endpoints import Endpoint, Process
from srtctl.core.process_registry import (
    ManagedProcess,
    NamedProcesses,
    ProcessRegistry,
    setup_signal_handlers,
    start_process_monitor,
)
from srtctl.core.runtime import RuntimeContext, get_hostname_ip, get_slurm_job_id
from srtctl.core.schema import SrtConfig
from srtctl.core.utils import start_srun_process, wait_for_health, wait_for_port
from srtctl.logging_utils import (
    CHECK,
    CROSS,
    GEAR,
    HOURGLASS,
    PACKAGE,
    ROCKET,
    WRENCH,
    get_logger,
    section,
    setup_logging,
)

logger = get_logger(__name__)


@dataclass
class SweepOrchestrator:
    """Main orchestrator for benchmark sweeps.

    Usage:
        config = load_config(config_path)  # Returns typed SrtConfig
        runtime = RuntimeContext.from_config(config, job_id)
        orchestrator = SweepOrchestrator(config, runtime)
        exit_code = orchestrator.run()
    """

    config: SrtConfig
    runtime: RuntimeContext

    @property
    def backend(self):
        """Access the backend config (implements BackendProtocol)."""
        return self.config.backend

    @functools.cached_property
    def endpoints(self) -> List[Endpoint]:
        """Compute endpoint allocation topology (cached).

        This is the single source of truth for endpoint assignments.
        """
        r = self.config.resources
        return self.backend.allocate_endpoints(
            num_prefill=r.num_prefill,
            num_decode=r.num_decode,
            num_agg=r.num_agg,
            gpus_per_prefill=r.gpus_per_prefill,
            gpus_per_decode=r.gpus_per_decode,
            gpus_per_agg=r.gpus_per_agg,
            gpus_per_node=r.gpus_per_node,
            available_nodes=self.runtime.nodes.worker,
        )

    @functools.cached_property
    def backend_processes(self) -> List[Process]:
        """Compute physical process topology from endpoints (cached)."""
        return self.backend.endpoints_to_processes(self.endpoints)

    def start_head_infrastructure(self, registry: ProcessRegistry) -> ManagedProcess:
        """Start NATS and etcd on the head node."""
        section("Starting head node infrastructure", ROCKET, logger)
        logger.info("Head node: %s", self.runtime.nodes.head)

        setup_script = Path(__file__).parent / "setup_head.py"
        if not setup_script.exists():
            raise RuntimeError(f"setup_head.py not found at {setup_script}")

        setup_script_container = Path("/tmp/setup_head.py")
        infra_log = self.runtime.log_dir / "log.out"

        cmd = [
            "python3", str(setup_script_container),
            "--name", self.config.name,
            "--log-dir", str(self.runtime.log_dir),
        ]

        mounts = dict(self.runtime.container_mounts)
        mounts[setup_script] = setup_script_container

        proc = start_srun_process(
            command=cmd,
            nodelist=[self.runtime.nodes.head],
            output=str(infra_log),
            container_image=str(self.runtime.container_image),
            container_mounts=mounts,
        )

        managed = ManagedProcess(
            name="head_infrastructure",
            popen=proc,
            log_file=infra_log,
            node=self.runtime.nodes.head,
            critical=True,
        )

        logger.info("Waiting for NATS (port 4222)...")
        if not wait_for_port(self.runtime.nodes.head, 4222, timeout=60):
            raise RuntimeError("NATS failed to start")
        logger.info("%s NATS is ready", CHECK)

        logger.info("Waiting for etcd (port 2379)...")
        if not wait_for_port(self.runtime.nodes.head, 2379, timeout=60):
            raise RuntimeError("etcd failed to start")
        logger.info("%s etcd is ready", CHECK)

        return managed

    def start_worker(self, process: Process, endpoint_processes: List[Process]) -> ManagedProcess:
        """Start a single worker process."""
        mode = process.endpoint_mode
        index = process.endpoint_index

        section(f"Starting {mode} worker {index} on {process.node}", WRENCH, logger)

        # Log and config files: {node}_{mode}_w{index}.out and {node}_config.json
        worker_log = self.runtime.log_dir / f"{process.node}_{mode}_w{index}.out"
        config_dump = self.runtime.log_dir / f"{process.node}_config.json"

        # Build command using backend's method
        cmd = self.backend.build_worker_command(
            process=process,
            endpoint_processes=endpoint_processes,
            runtime=self.runtime,
            use_sglang_router=self.config.frontend.use_sglang_router,
            dump_config_path=config_dump,
        )

        # Environment variables
        env_to_set = {
            "HEAD_NODE_IP": self.runtime.head_node_ip,
            "ETCD_ENDPOINTS": f"http://{self.runtime.nodes.head}:2379",
            "NATS_SERVER": f"nats://{self.runtime.nodes.head}:4222",
            "DYN_SYSTEM_PORT": str(process.sys_port),
        }

        # Add mode-specific environment variables from backend
        env_to_set.update(self.backend.get_environment_for_mode(mode))

        # Add config environment variables
        env_to_set.update(self.runtime.environment)

        # Set CUDA_VISIBLE_DEVICES if not using all GPUs
        if len(process.gpu_indices) < self.runtime.gpus_per_node:
            env_to_set["CUDA_VISIBLE_DEVICES"] = process.cuda_visible_devices

        logger.info("Command: %s", shlex.join(cmd))
        logger.info("Log: %s", worker_log)

        # Install dynamo when not using sglang router (required for dynamo.sglang module)
        bash_preamble = None
        if not self.config.frontend.use_sglang_router:
            bash_preamble = (
                "echo 'Installing dynamo...' && "
                "pip install --quiet ai-dynamo-runtime==0.7.0 ai-dynamo==0.7.0 && "
                "echo 'Dynamo installed'"
            )

        proc = start_srun_process(
            command=cmd,
            nodelist=[process.node],
            output=str(worker_log),
            container_image=str(self.runtime.container_image),
            container_mounts=self.runtime.container_mounts,
            env_to_set=env_to_set,
            bash_preamble=bash_preamble,
        )

        return ManagedProcess(
            name=f"{mode}_{index}_{process.node}",
            popen=proc,
            log_file=worker_log,
            node=process.node,
            critical=True,
        )

    def start_all_workers(self) -> NamedProcesses:
        """Start all backend workers."""
        section("Starting backend workers", PACKAGE, logger)

        from collections import defaultdict
        grouped: Dict[tuple, List[Process]] = defaultdict(list)
        for process in self.backend_processes:
            key = (process.endpoint_mode, process.endpoint_index)
            grouped[key].append(process)

        result: NamedProcesses = {}
        for endpoint_key, endpoint_processes in grouped.items():
            for process in endpoint_processes:
                managed = self.start_worker(process, endpoint_processes)
                result[managed.name] = managed

        logger.info("Started %d worker processes", len(result))
        return result

    def start_frontend(self, registry: ProcessRegistry) -> Optional[ManagedProcess]:
        """Start the frontend."""
        section("Starting frontend", PACKAGE, logger)

        if self.config.frontend.use_sglang_router:
            return self._start_sglang_router()
        else:
            return self._start_dynamo_frontend()

    def _start_dynamo_frontend(self) -> ManagedProcess:
        """Start dynamo frontend on the head node."""
        logger.info("Starting dynamo frontend on %s", self.runtime.nodes.head)

        frontend_log = self.runtime.log_dir / f"{self.runtime.nodes.head}_frontend_0.out"
        cmd = ["python3", "-m", "dynamo.frontend", "--http-port=8000"]
        cmd.extend(self.config.frontend.get_router_args_list())

        env_to_set = {
            "ETCD_ENDPOINTS": f"http://{self.runtime.nodes.head}:2379",
            "NATS_SERVER": f"nats://{self.runtime.nodes.head}:4222",
        }

        # Install dynamo before starting frontend
        bash_preamble = (
            "echo 'Installing dynamo...' && "
            "pip install --quiet ai-dynamo-runtime==0.7.0 ai-dynamo==0.7.0 && "
            "echo 'Dynamo installed'"
        )

        proc = start_srun_process(
            command=cmd,
            nodelist=[self.runtime.nodes.head],
            output=str(frontend_log),
            container_image=str(self.runtime.container_image),
            container_mounts=self.runtime.container_mounts,
            env_to_set=env_to_set,
            bash_preamble=bash_preamble,
        )

        return ManagedProcess(
            name="frontend",
            popen=proc,
            log_file=frontend_log,
            node=self.runtime.nodes.head,
            critical=True,
        )

    def _start_sglang_router(self) -> ManagedProcess:
        """Start sglang-router on the head node."""
        logger.info("Starting sglang-router on %s", self.runtime.nodes.head)

        router_log = self.runtime.log_dir / f"{self.runtime.nodes.head}_router.out"

        # Collect prefill and decode leader IPs
        prefill_ips = []
        decode_ips = []
        for endpoint in self.endpoints:
            leader_ip = get_hostname_ip(endpoint.leader_node)
            if endpoint.mode == "prefill":
                prefill_ips.append(leader_ip)
            elif endpoint.mode == "decode":
                decode_ips.append(leader_ip)

        cmd = ["python", "-m", "sglang_router.launch_router", "--pd-disaggregation"]

        server_port = 30000
        bootstrap_port = 30001

        for ip in prefill_ips:
            cmd.extend(["--prefill", f"http://{ip}:{server_port}", str(bootstrap_port)])
        for ip in decode_ips:
            cmd.extend(["--decode", f"http://{ip}:{server_port}"])

        cmd.extend(["--host", "0.0.0.0", "--port", "8000"])
        cmd.extend(self.config.frontend.get_router_args_list())

        logger.info("Router command: %s", shlex.join(cmd))

        proc = start_srun_process(
            command=cmd,
            nodelist=[self.runtime.nodes.head],
            output=str(router_log),
            container_image=str(self.runtime.container_image),
            container_mounts=self.runtime.container_mounts,
        )

        return ManagedProcess(
            name="sglang_router",
            popen=proc,
            log_file=router_log,
            node=self.runtime.nodes.head,
            critical=True,
        )

    def run_benchmark(self, registry: ProcessRegistry, stop_event: threading.Event) -> int:
        """Run the benchmark."""
        section("Running benchmark", ROCKET, logger)

        r = self.config.resources
        num_workers = r.num_prefill + r.num_decode + r.num_agg

        logger.info("Waiting for server health (expecting %d workers)...", num_workers)

        hc = self.config.health_check
        if not wait_for_health(
            self.runtime.nodes.head,
            8000,
            max_attempts=hc.max_attempts,
            interval=float(hc.interval_seconds),
            expected_workers=num_workers,
            stop_event=stop_event,
        ):
            logger.error("%s Server did not become healthy", CROSS)
            return 1

        logger.info("%s Server is healthy", CHECK)

        if self.config.benchmark.type == "manual":
            logger.info("Benchmark type is 'manual' - server is ready for testing")
            logger.info("Frontend URL: http://%s:8000", self.runtime.nodes.head)
            logger.info("Press Ctrl+C to stop the job")

            while not stop_event.is_set():
                if registry.check_failures():
                    logger.error("Worker failure detected during manual mode")
                    return 1
                time.sleep(5)
            return 0

        logger.info("Benchmark type '%s' - running benchmark", self.config.benchmark.type)
        # TODO: Implement actual benchmark running for sa-bench, etc.
        time.sleep(10)
        return 0

    def run(self) -> int:
        """Run the complete sweep."""
        section("Sweep Orchestrator", ROCKET, logger)
        logger.info("Job ID: %s", self.runtime.job_id)
        logger.info("Run name: %s", self.runtime.run_name)
        logger.info("Config: %s", self.config.name)
        logger.info("Head node: %s", self.runtime.nodes.head)
        logger.info("Worker nodes: %s", ", ".join(self.runtime.nodes.worker))

        registry = ProcessRegistry(job_id=self.runtime.job_id)
        stop_event = threading.Event()
        setup_signal_handlers(stop_event, registry)
        start_process_monitor(stop_event, registry)

        exit_code = 1

        try:
            head_proc = self.start_head_infrastructure(registry)
            registry.add_process(head_proc)

            worker_procs = self.start_all_workers()
            registry.add_processes(worker_procs)

            frontend_proc = self.start_frontend(registry)
            if frontend_proc:
                registry.add_process(frontend_proc)

            exit_code = self.run_benchmark(registry, stop_event)

        except Exception as e:
            logger.exception("Error during sweep: %s", e)
            exit_code = 1

        finally:
            section("Cleanup", PACKAGE, logger)
            stop_event.set()
            registry.cleanup()
            if exit_code != 0:
                registry.print_failure_details()

        return exit_code


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run benchmark sweep")
    parser.add_argument("config", type=str, help="Path to YAML configuration file")
    args = parser.parse_args()

    setup_logging()

    try:
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error("Config file not found: %s", config_path)
            sys.exit(1)

        config = load_config(config_path)

        job_id = get_slurm_job_id()
        if not job_id:
            logger.error("Not running in SLURM (SLURM_JOB_ID not set)")
            sys.exit(1)

        runtime = RuntimeContext.from_config(config, job_id)
        orchestrator = SweepOrchestrator(config=config, runtime=runtime)
        exit_code = orchestrator.run()

        sys.exit(exit_code)

    except Exception as e:
        logger.exception("Fatal error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

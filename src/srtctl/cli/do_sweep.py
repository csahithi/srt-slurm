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
import logging
import shlex
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from srtctl.benchmarks.base import BenchmarkRunner

from srtctl.core.config import load_config
from srtctl.core.health import wait_for_model, wait_for_port
from srtctl.core.processes import (
    ManagedProcess,
    NamedProcesses,
    ProcessRegistry,
    setup_signal_handlers,
    start_process_monitor,
)
from srtctl.core.runtime import RuntimeContext
from srtctl.core.schema import SrtConfig
from srtctl.core.slurm import get_hostname_ip, get_slurm_job_id, start_srun_process
from srtctl.core.topology import Endpoint, Process
from srtctl.logging_utils import setup_logging

logger = logging.getLogger(__name__)


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
    def endpoints(self) -> list[Endpoint]:
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
    def backend_processes(self) -> list[Process]:
        """Compute physical process topology from endpoints (cached)."""
        return self.backend.endpoints_to_processes(self.endpoints)

    def _build_worker_preamble(self) -> str | None:
        """Build bash preamble for worker processes.

        Runs (in order):
        1. Custom setup script from /configs/ (if config.setup_script set)
        2. Dynamo installation (if not using sglang router)
        """
        parts = []

        # 1. Custom setup script (runs first)
        if self.config.setup_script:
            script_path = f"/configs/{self.config.setup_script}"
            parts.append(
                f"echo 'Running setup script: {script_path}' && "
                f"if [ -f '{script_path}' ]; then bash '{script_path}'; else echo 'WARNING: {script_path} not found'; fi"
            )

        # 2. Dynamo installation (required for dynamo.sglang when not using sglang router and not profiling)
        # When profiling is enabled, we use sglang.launch_server directly (no dynamo)
        if not self.config.frontend.use_sglang_router and not self.config.profiling.enabled:
            parts.append(
                "echo 'Installing dynamo...' && "
                "pip install --quiet ai-dynamo-runtime==0.7.0 ai-dynamo==0.7.0 && "
                "echo 'Dynamo installed'"
            )

        if not parts:
            return None

        return " && ".join(parts)

    def start_head_infrastructure(self, registry: ProcessRegistry) -> ManagedProcess:
        """Start NATS and etcd on the head node."""
        logger.info("Starting head node infrastructure")
        logger.info("Head node: %s", self.runtime.nodes.head)

        setup_script = Path(__file__).parent / "setup_head.py"
        if not setup_script.exists():
            raise RuntimeError(f"setup_head.py not found at {setup_script}")

        setup_script_container = Path("/tmp/setup_head.py")
        infra_log = self.runtime.log_dir / "log.out"

        cmd = [
            "python3",
            str(setup_script_container),
            "--name",
            self.config.name,
            "--log-dir",
            str(self.runtime.log_dir),
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
        logger.info("NATS is ready")

        logger.info("Waiting for etcd (port 2379)...")
        if not wait_for_port(self.runtime.nodes.head, 2379, timeout=60):
            raise RuntimeError("etcd failed to start")
        logger.info("etcd is ready")

        return managed

    def start_worker(self, process: Process, endpoint_processes: list[Process]) -> ManagedProcess:
        """Start a single worker process."""
        mode = process.endpoint_mode
        index = process.endpoint_index

        logger.info("Starting %s worker %d on %s", mode, index, process.node)

        # Log and config files
        worker_log = self.runtime.log_dir / f"{process.node}_{mode}_w{index}.out"
        config_dump = self.runtime.log_dir / f"{process.node}_config.json"

        # Profiling setup
        profiling = self.config.profiling
        nsys_prefix = None
        if profiling.is_nsys:
            nsys_output = str(self.runtime.log_dir / f"{process.node}_{mode}_w{index}_profile")
            nsys_prefix = profiling.get_nsys_prefix(nsys_output)

        # Build command using backend's method
        cmd = self.backend.build_worker_command(
            process=process,
            endpoint_processes=endpoint_processes,
            runtime=self.runtime,
            use_sglang_router=self.config.frontend.use_sglang_router,
            profiling_enabled=profiling.enabled,
            nsys_prefix=nsys_prefix,
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

        # Add profiling environment variables
        if profiling.enabled:
            profile_dir = str(self.runtime.log_dir / "profiles")
            env_to_set.update(profiling.get_env_vars(mode, profile_dir))

        # Set CUDA_VISIBLE_DEVICES if not using all GPUs
        if len(process.gpu_indices) < self.runtime.gpus_per_node:
            env_to_set["CUDA_VISIBLE_DEVICES"] = process.cuda_visible_devices

        # Log env vars in the format: VAR=value VAR2=value2
        env_str = " ".join(f"{k}={v}" for k, v in sorted(env_to_set.items()))
        logger.info("Env: %s", env_str)
        logger.info("Command: %s", shlex.join(cmd))
        logger.info("Log: %s", worker_log)
        if profiling.enabled:
            logger.info("Profiling: %s mode", profiling.type)

        # Build bash preamble (setup script + dynamo install)
        bash_preamble = self._build_worker_preamble()

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
        logger.info("Starting backend workers")

        from collections import defaultdict

        grouped: dict[tuple, list[Process]] = defaultdict(list)
        for process in self.backend_processes:
            key = (process.endpoint_mode, process.endpoint_index)
            grouped[key].append(process)

        result: NamedProcesses = {}
        for _endpoint_key, endpoint_processes in grouped.items():
            for process in endpoint_processes:
                managed = self.start_worker(process, endpoint_processes)
                result[managed.name] = managed

        logger.info("Started %d worker processes", len(result))
        return result

    def start_frontend(self, registry: ProcessRegistry) -> ManagedProcess | None:
        """Start the frontend."""
        logger.info("Starting frontend")

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

        # Use the same preamble (setup script + dynamo install) as workers
        bash_preamble = self._build_worker_preamble()

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

        # Collect prefill and decode leader info from processes
        prefill_leaders: list[tuple[str, int, int]] = []  # (ip, http_port, bootstrap_port)
        decode_leaders: list[tuple[str, int]] = []  # (ip, http_port)

        for process in self.backend_processes:
            if not process.is_leader:
                continue
            leader_ip = get_hostname_ip(process.node)
            if process.endpoint_mode == "prefill":
                assert process.bootstrap_port is not None
                prefill_leaders.append((leader_ip, process.http_port, process.bootstrap_port))
            elif process.endpoint_mode == "decode":
                decode_leaders.append((leader_ip, process.http_port))

        cmd = ["python", "-m", "sglang_router.launch_router", "--pd-disaggregation"]

        for ip, http_port, bootstrap_port in prefill_leaders:
            cmd.extend(["--prefill", f"http://{ip}:{http_port}", str(bootstrap_port)])
        for ip, http_port in decode_leaders:
            cmd.extend(["--decode", f"http://{ip}:{http_port}"])

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
        logger.info("Running benchmark")

        r = self.config.resources
        num_workers = r.num_prefill + r.num_decode + r.num_agg

        # Build descriptive worker count string
        worker_desc = f"{r.num_agg} agg" if r.num_agg > 0 else f"{r.num_prefill}P + {r.num_decode}D"

        logger.info("Waiting for server health (expecting %d workers: %s)...", num_workers, worker_desc)

        # For aggregated mode: expect 0 prefill, N decode (backend workers count as decode)
        # For disaggregated mode: expect N prefill, M decode
        if r.num_agg > 0:
            n_prefill = 0
            n_decode = r.num_agg
        else:
            n_prefill = r.num_prefill
            n_decode = r.num_decode

        hc = self.config.health_check
        if not wait_for_model(
            host=self.runtime.nodes.head,
            port=8000,
            n_prefill=n_prefill,
            n_decode=n_decode,
            poll_interval=float(hc.interval_seconds),
            timeout=float(hc.max_attempts * hc.interval_seconds),
            report_every=60.0,
            use_sglang_router=self.config.frontend.use_sglang_router,
            stop_event=stop_event,
        ):
            logger.error("Server did not become healthy")
            return 1

        logger.info("Server is healthy")

        # Auto-select profiling benchmark when profiling is enabled
        benchmark_type = self.config.benchmark.type
        if self.config.profiling.enabled:
            if benchmark_type != "profiling":
                logger.info(
                    "Profiling enabled (type=%s) - automatically using 'profiling' benchmark",
                    self.config.profiling.type,
                )
                logger.info(
                    "Profiling config: isl=%s, osl=%s, concurrency=%s",
                    self.config.profiling.isl,
                    self.config.profiling.osl,
                    self.config.profiling.concurrency,
                )
            benchmark_type = "profiling"

        if benchmark_type == "manual":
            logger.info("Benchmark type is 'manual' - server is ready for testing")
            logger.info("Frontend URL: http://%s:8000", self.runtime.nodes.head)
            logger.info("Press Ctrl+C to stop the job")

            while not stop_event.is_set():
                if registry.check_failures():
                    logger.error("Worker failure detected during manual mode")
                    return 1
                time.sleep(5)
            return 0

        # Get the appropriate benchmark runner
        from srtctl.benchmarks import get_runner

        try:
            runner = get_runner(benchmark_type)
        except ValueError as e:
            logger.error("%s", e)
            return 1

        # Validate config
        errors = runner.validate_config(self.config)
        if errors:
            for error in errors:
                logger.error("Config error: %s", error)
            return 1

        logger.info("Running %s benchmark", runner.name)

        # Run the benchmark script
        benchmark_log = self.runtime.log_dir / "benchmark.out"
        exit_code = self._run_benchmark_script(runner, benchmark_log, stop_event)

        if exit_code != 0:
            logger.error("Benchmark failed with exit code %d", exit_code)
        else:
            logger.info("Benchmark completed successfully")

        return exit_code

    def _run_benchmark_script(
        self,
        runner: "BenchmarkRunner",
        log_file: Path,
        stop_event: threading.Event,
    ) -> int:
        """Run the actual benchmark script."""

        cmd = runner.build_command(self.config, self.runtime)
        env_to_set = self._get_benchmark_profiling_env(runner)

        logger.info("Script: %s", runner.script_path)
        logger.info("Command: %s", shlex.join(cmd))
        logger.info("Log: %s", log_file)

        proc = start_srun_process(
            command=cmd,
            nodelist=[self.runtime.nodes.head],
            output=str(log_file),
            container_image=str(self.runtime.container_image),
            container_mounts=self.runtime.container_mounts,
            env_to_set=env_to_set,
        )

        # Wait for benchmark to complete
        while proc.poll() is None:
            if stop_event.is_set():
                logger.info("Stop requested, terminating benchmark")
                proc.terminate()
                return 1
            time.sleep(1)

        return proc.returncode or 0

    def _get_benchmark_profiling_env(self, runner: "BenchmarkRunner") -> dict[str, str]:
        """Get environment variables for the benchmark script."""
        env: dict[str, str] = {}

        # Add profiling-specific env vars
        if runner.name == "Profiling" and self.config.profiling.enabled:
            p = self.config.profiling

            # Traffic generator params
            if p.isl is not None:
                env["PROFILE_ISL"] = str(p.isl)
            if p.osl is not None:
                env["PROFILE_OSL"] = str(p.osl)
            if p.concurrency is not None:
                env["PROFILE_CONCURRENCY"] = str(p.concurrency)

            # Model name
            env["PROFILE_MODEL_NAME"] = self.config.served_model_name

            # Head node for traffic
            env["HEAD_NODE"] = self.runtime.nodes.head
            env["HEAD_PORT"] = str(self.runtime.frontend_port)

            # Collect worker leader IPs by mode
            prefill_ips = []
            decode_ips = []
            agg_ips = []

            for endpoint in self.endpoints:
                leader_ip = get_hostname_ip(endpoint.leader_node)
                if endpoint.mode == "prefill":
                    prefill_ips.append(leader_ip)
                elif endpoint.mode == "decode":
                    decode_ips.append(leader_ip)
                elif endpoint.mode == "agg":
                    agg_ips.append(leader_ip)

            if prefill_ips:
                env["PROFILE_PREFILL_IPS"] = ",".join(prefill_ips)
            if decode_ips:
                env["PROFILE_DECODE_IPS"] = ",".join(decode_ips)
            if agg_ips:
                env["PROFILE_AGG_IPS"] = ",".join(agg_ips)

            # Phase-specific step configs
            if p.prefill:
                if p.prefill.start_step is not None:
                    env["PROFILE_PREFILL_START_STEP"] = str(p.prefill.start_step)
                if p.prefill.stop_step is not None:
                    env["PROFILE_PREFILL_STOP_STEP"] = str(p.prefill.stop_step)
            if p.decode:
                if p.decode.start_step is not None:
                    env["PROFILE_DECODE_START_STEP"] = str(p.decode.start_step)
                if p.decode.stop_step is not None:
                    env["PROFILE_DECODE_STOP_STEP"] = str(p.decode.stop_step)
            if p.aggregated:
                if p.aggregated.start_step is not None:
                    env["PROFILE_AGG_START_STEP"] = str(p.aggregated.start_step)
                if p.aggregated.stop_step is not None:
                    env["PROFILE_AGG_STOP_STEP"] = str(p.aggregated.stop_step)

            # Torch profiler directory
            if p.is_torch:
                env["SGLANG_TORCH_PROFILER_DIR"] = str(self.runtime.log_dir / "profiles")

            # The profile.sh script only generates traffic when PROFILING_MODE=prefill
            env["PROFILING_MODE"] = "prefill"

        return env

    def _print_connection_info(self) -> None:
        """Print srun commands for connecting to nodes."""
        container_args = f"--container-image={self.runtime.container_image}"
        mounts_str = ",".join(f"{src}:{dst}" for src, dst in self.runtime.container_mounts.items())
        if mounts_str:
            container_args += f" --container-mounts={mounts_str}"

        logger.info("")
        logger.info("=" * 60)
        logger.info("Connection Commands")
        logger.info("=" * 60)
        logger.info("Frontend URL: http://%s:8000", self.runtime.nodes.head)
        logger.info("")
        logger.info("To connect to head node (%s):", self.runtime.nodes.head)
        logger.info(
            "  srun %s --jobid %s -w %s --overlap --pty bash",
            container_args,
            self.runtime.job_id,
            self.runtime.nodes.head,
        )

        # Print worker node connection commands
        for node in self.runtime.nodes.worker:
            if node != self.runtime.nodes.head:
                logger.info("")
                logger.info("To connect to worker node (%s):", node)
                logger.info(
                    "  srun %s --jobid %s -w %s --overlap --pty bash",
                    container_args,
                    self.runtime.job_id,
                    node,
                )

        logger.info("=" * 60)
        logger.info("")

    def run(self) -> int:
        """Run the complete sweep."""
        logger.info("Sweep Orchestrator")
        logger.info("Job ID: %s", self.runtime.job_id)
        logger.info("Run name: %s", self.runtime.run_name)
        logger.info("Config: %s", self.config.name)
        logger.info("Head node: %s", self.runtime.nodes.head)
        logger.info("Worker nodes: %s", ", ".join(self.runtime.nodes.worker))
        if self.config.profiling.enabled:
            logger.info(
                "Profiling: %s (isl=%s, osl=%s, concurrency=%s)",
                self.config.profiling.type,
                self.config.profiling.isl,
                self.config.profiling.osl,
                self.config.profiling.concurrency,
            )

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

            self._print_connection_info()

            exit_code = self.run_benchmark(registry, stop_event)

        except Exception as e:
            logger.exception("Error during sweep: %s", e)
            exit_code = 1

        finally:
            logger.info("Cleanup")
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

        # Type narrowing: job_id is str after the check above
        assert job_id is not None
        runtime = RuntimeContext.from_config(config, job_id)
        orchestrator = SweepOrchestrator(config=config, runtime=runtime)
        exit_code = orchestrator.run()

        sys.exit(exit_code)

    except Exception as e:
        logger.exception("Fatal error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

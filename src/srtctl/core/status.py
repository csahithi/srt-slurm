# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Fire-and-forget status reporter for external job tracking.

This module provides optional status reporting to an external API endpoint.
If the endpoint is not configured or unreachable, operations silently continue.
The API contract is defined in docs/status-api-spec.md.

Environment Variables:
    JOB_STATUS_REPORT_ENDPOINT: Base URL of status API (e.g., https://status.example.com)
    JOB_STATUS_REPORT_CLUSTER: Cluster name for job metadata
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig

logger = logging.getLogger(__name__)

# Environment variable names
STATUS_API_ENV = "JOB_STATUS_REPORT_ENDPOINT"
CLUSTER_ENV = "JOB_STATUS_REPORT_CLUSTER"


class JobStage(str, Enum):
    """Job execution stages matching do_sweep.py flow."""

    STARTING = "starting"
    HEAD_INFRASTRUCTURE = "head_infrastructure"
    WORKERS = "workers"
    FRONTEND = "frontend"
    BENCHMARK = "benchmark"
    CLEANUP = "cleanup"


class JobStatus(str, Enum):
    """Job status values."""

    SUBMITTED = "submitted"
    STARTING = "starting"
    HEAD_INFRA_READY = "head_ready"
    WORKERS_STARTING = "workers_starting"
    WORKERS_READY = "workers_ready"
    FRONTEND_READY = "frontend_ready"
    BENCHMARK_RUNNING = "benchmark"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class StatusReporter:
    """Fire-and-forget status reporter.

    Reports job status to an external API if JOB_STATUS_REPORT_ENDPOINT is set.
    All operations are non-blocking and failures are silently logged.

    Usage:
        reporter = StatusReporter.from_env(job_id="12345")
        reporter.report(JobStatus.WORKERS_READY, stage=JobStage.WORKERS)
    """

    job_id: str
    api_endpoint: str | None = None
    cluster: str | None = None
    timeout: float = 5.0

    @classmethod
    def from_env(cls, job_id: str) -> "StatusReporter":
        """Create reporter from environment variables."""
        endpoint = os.environ.get(STATUS_API_ENV)
        cluster = os.environ.get(CLUSTER_ENV)

        if endpoint:
            # Strip trailing slash for consistency
            endpoint = endpoint.rstrip("/")
            logger.info("Status reporting enabled: %s", endpoint)

        return cls(job_id=job_id, api_endpoint=endpoint, cluster=cluster)

    @property
    def enabled(self) -> bool:
        """Check if reporting is enabled."""
        return self.api_endpoint is not None

    def _now_iso(self) -> str:
        """Get current UTC time in ISO8601 format."""
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def report(
        self,
        status: JobStatus,
        stage: JobStage | None = None,
        message: str | None = None,
    ) -> bool:
        """Report status update (fire-and-forget).

        Args:
            status: New job status
            stage: Current execution stage
            message: Optional human-readable message

        Returns:
            True if reported successfully, False otherwise
        """
        if not self.enabled:
            return False

        try:
            payload: dict = {
                "status": status.value,
                "updated_at": self._now_iso(),
            }
            if stage:
                payload["stage"] = stage.value
            if message:
                payload["message"] = message

            url = f"{self.api_endpoint}/api/jobs/{self.job_id}"
            response = requests.put(url, json=payload, timeout=self.timeout)

            if response.status_code == 200:
                logger.debug("Status reported: %s", status.value)
                return True
            else:
                logger.debug("Status report failed: HTTP %d", response.status_code)
                return False

        except requests.exceptions.RequestException as e:
            logger.debug("Status report error (ignored): %s", e)
            return False

    def report_started(self, config: "SrtConfig", runtime: "RuntimeContext") -> bool:
        """Report job started with initial metadata.

        Args:
            config: Job configuration
            runtime: Runtime context with computed values

        Returns:
            True if reported successfully, False otherwise
        """
        if not self.enabled:
            return False

        try:
            metadata = {
                "model": {
                    "path": str(config.model.path),
                    "precision": config.model.precision,
                },
                "resources": {
                    "gpu_type": config.resources.gpu_type,
                    "gpus_per_node": config.resources.gpus_per_node,
                    "prefill_workers": config.resources.num_prefill,
                    "decode_workers": config.resources.num_decode,
                    "agg_workers": config.resources.num_agg,
                },
                "benchmark": {
                    "type": config.benchmark.type,
                },
                "backend_type": config.backend_type,
                "frontend_type": config.frontend.type,
                "head_node": runtime.nodes.head,
            }

            payload = {
                "status": JobStatus.STARTING.value,
                "stage": JobStage.STARTING.value,
                "message": f"Job started on {runtime.nodes.head}",
                "started_at": self._now_iso(),
                "updated_at": self._now_iso(),
                "metadata": metadata,
            }

            url = f"{self.api_endpoint}/api/jobs/{self.job_id}"
            response = requests.put(url, json=payload, timeout=self.timeout)
            return response.status_code == 200

        except requests.exceptions.RequestException as e:
            logger.debug("Status report error (ignored): %s", e)
            return False

    def report_completed(self, exit_code: int) -> bool:
        """Report job completed with exit code.

        Args:
            exit_code: Process exit code (0 = success)

        Returns:
            True if reported successfully, False otherwise
        """
        if not self.enabled:
            return False

        try:
            status = JobStatus.COMPLETED if exit_code == 0 else JobStatus.FAILED
            message = "Benchmark completed successfully" if exit_code == 0 else f"Job failed with exit code {exit_code}"

            payload = {
                "status": status.value,
                "stage": JobStage.CLEANUP.value,
                "message": message,
                "completed_at": self._now_iso(),
                "updated_at": self._now_iso(),
                "exit_code": exit_code,
            }

            url = f"{self.api_endpoint}/api/jobs/{self.job_id}"
            response = requests.put(url, json=payload, timeout=self.timeout)
            return response.status_code == 200

        except requests.exceptions.RequestException as e:
            logger.debug("Status report error (ignored): %s", e)
            return False


def create_job_record(
    job_id: str,
    job_name: str,
    recipe: str | None = None,
    metadata: dict | None = None,
) -> bool:
    """Create initial job record in status API (called at submission time).

    This is a standalone function used by submit.py before the job starts.

    Args:
        job_id: SLURM job ID
        job_name: Job/config name
        recipe: Path to recipe file (optional)
        metadata: Job metadata dict (optional)

    Returns:
        True if created successfully, False otherwise
    """
    api_endpoint = os.environ.get(STATUS_API_ENV)
    if not api_endpoint:
        return False

    api_endpoint = api_endpoint.rstrip("/")
    cluster = os.environ.get(CLUSTER_ENV)

    try:
        payload: dict = {
            "job_id": job_id,
            "job_name": job_name,
            "submitted_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        if cluster:
            payload["cluster"] = cluster
        if recipe:
            payload["recipe"] = recipe
        if metadata:
            payload["metadata"] = metadata

        url = f"{api_endpoint}/api/jobs"
        response = requests.post(url, json=payload, timeout=5.0)

        if response.status_code == 201:
            logger.debug("Job record created: %s", job_id)
            return True
        else:
            logger.debug("Job record creation failed: HTTP %d", response.status_code)
            return False

    except requests.exceptions.RequestException as e:
        logger.debug("Job record creation error (ignored): %s", e)
        return False

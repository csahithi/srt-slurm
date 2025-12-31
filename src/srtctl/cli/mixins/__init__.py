# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Stage mixins for SweepOrchestrator.

Each mixin handles one stage of the sweep orchestration:
- WorkerStageMixin: Backend worker process startup
- FrontendStageMixin: Frontend/nginx orchestration
- BenchmarkStageMixin: Benchmark execution
- MetricsStageMixin: Prometheus metrics collection
"""

from srtctl.cli.mixins.benchmark_stage import BenchmarkStageMixin
from srtctl.cli.mixins.frontend_stage import FrontendStageMixin
from srtctl.cli.mixins.metrics_stage import MetricsStageMixin
from srtctl.cli.mixins.worker_stage import WorkerStageMixin

__all__ = [
    "WorkerStageMixin",
    "FrontendStageMixin",
    "BenchmarkStageMixin",
    "MetricsStageMixin",
]

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Rollup dataclasses for experiment data consolidation.

This module provides dataclasses for:
- RollupResult: Single benchmark result at one concurrency level
- RollupSummary: Complete experiment summary
- NodeRollup: Single worker node metrics
- NodesSummary: Summary of all worker nodes
- EnvironmentConfig: Environment variables and engine config
- LaunchCommandRollup: Parsed launch command information
"""

from srtctl.cli.mixins.rollup.models import (
    EnvironmentConfig,
    LaunchCommandRollup,
    NodeRollup,
    NodesSummary,
    RollupResult,
    RollupSummary,
)

__all__ = [
    "RollupResult",
    "RollupSummary",
    "NodeRollup",
    "NodesSummary",
    "EnvironmentConfig",
    "LaunchCommandRollup",
]

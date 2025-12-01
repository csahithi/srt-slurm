# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker setup module for SGLang distributed serving."""

from .command import build_sglang_command_from_yaml, get_gpu_command, install_dynamo_wheels
from .environment import setup_env
from .infrastructure import setup_frontend_worker, setup_head_prefill_node, setup_nginx_worker, setup_sglang_router
from .utils import setup_logging, wait_for_etcd
from .worker import setup_aggregated_worker, setup_decode_worker, setup_prefill_worker

__all__ = [
    # Command building
    "build_sglang_command_from_yaml",
    "get_gpu_command",
    "install_dynamo_wheels",
    # Environment
    "setup_env",
    # Infrastructure
    "setup_frontend_worker",
    "setup_head_prefill_node",
    "setup_nginx_worker",
    "setup_sglang_router",
    # Utils
    "setup_logging",
    "wait_for_etcd",
    # Workers
    "setup_aggregated_worker",
    "setup_decode_worker",
    "setup_prefill_worker",
]

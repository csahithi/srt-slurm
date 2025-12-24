# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Backend implementations for different LLM serving frameworks.

Supported backends:
- SGLang: Full support with prefill/decode disaggregation
"""

from .base import BackendProtocol, BackendType
from .sglang import SGLangProtocol, SGLangServerConfig

# Union type for all backend configs
BackendConfig = SGLangProtocol

__all__ = [
    # Base types
    "BackendProtocol",
    "BackendType",
    "BackendConfig",
    # SGLang
    "SGLangProtocol",
    "SGLangServerConfig",
]

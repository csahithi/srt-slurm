#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Config loading and resolution with srtslurm.yaml integration.

This module provides:
- load_config(): Load YAML config, apply cluster defaults, return typed SrtConfig
- get_srtslurm_setting(): Get cluster-wide settings
"""

import copy
import logging
from pathlib import Path
from typing import Any

import yaml

from .schema import ClusterConfig, SrtConfig

logger = logging.getLogger(__name__)


def load_cluster_config() -> dict[str, Any] | None:
    """
    Load cluster configuration from srtslurm.yaml if it exists.

    Searches for srtslurm.yaml in:
    1. Current working directory
    2. Parent directories up to 3 levels

    Returns None if file doesn't exist (graceful degradation).
    """
    # Search paths
    search_paths = [
        Path.cwd() / "srtslurm.yaml",
        Path.cwd().parent / "srtslurm.yaml",
        Path.cwd().parent.parent / "srtslurm.yaml",
    ]

    cluster_config_path = None
    for path in search_paths:
        if path.exists():
            cluster_config_path = path
            break

    if not cluster_config_path:
        logger.debug("No srtslurm.yaml found - using config as-is")
        return None

    try:
        with open(cluster_config_path) as f:
            raw_config = yaml.safe_load(f)

        # Validate with marshmallow schema
        schema = ClusterConfig.Schema()
        validated = schema.load(raw_config)
        logger.debug(f"Loaded cluster config from {cluster_config_path}")

        # Dump back to dict for compatibility
        return schema.dump(validated)
    except Exception as e:
        logger.warning(f"Failed to load or validate srtslurm.yaml: {e}")
        return None


def merge_cli_overrides(config_dict: dict[str, Any], model_overrides: dict[str, Any] | None = None, slurm_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Merge CLI overrides into config dict.

    CLI overrides take highest priority - they override both YAML config and cluster defaults.

    Args:
        config_dict: Base config dict from YAML
        model_overrides: Model overrides dict (e.g., {"path": "...", "container": "...", "precision": "..."})
        slurm_overrides: SLURM overrides dict (e.g., {"account": "...", "partition": "...", "time_limit": "..."})

    Returns:
        Config dict with overrides merged in
    """
    config = copy.deepcopy(config_dict)

    if model_overrides:
        config.setdefault("model", {}).update(model_overrides)
        logger.debug(f"Applied model overrides: {model_overrides}")

    if slurm_overrides:
        config.setdefault("slurm", {}).update(slurm_overrides)
        logger.debug(f"Applied SLURM overrides: {slurm_overrides}")

    return config


def resolve_config_with_defaults(user_config: dict[str, Any], cluster_config: dict[str, Any] | None) -> dict[str, Any]:
    """
    Resolve user config by applying cluster defaults and aliases.

    This applies:
    1. Default SLURM settings (account, partition, time_limit)
    2. Model path alias resolution
    3. Container alias resolution

    Args:
        user_config: User's YAML config as dict
        cluster_config: Cluster defaults from srtslurm.yaml (or None)

    Returns:
        Resolved config dict with all defaults applied
    """
    # Deep copy to avoid mutating original
    config = copy.deepcopy(user_config)

    if cluster_config is None:
        return config

    # Apply SLURM defaults
    slurm = config.setdefault("slurm", {})
    if "account" not in slurm and cluster_config.get("default_account"):
        slurm["account"] = cluster_config["default_account"]
        logger.debug(f"Applied default account: {slurm['account']}")

    if "partition" not in slurm and cluster_config.get("default_partition"):
        slurm["partition"] = cluster_config["default_partition"]
        logger.debug(f"Applied default partition: {slurm['partition']}")

    if "time_limit" not in slurm and cluster_config.get("default_time_limit"):
        slurm["time_limit"] = cluster_config["default_time_limit"]
        logger.debug(f"Applied default time_limit: {slurm['time_limit']}")

    # Resolve model path alias
    model = config.get("model", {})
    model_path = model.get("path", "")

    model_paths = cluster_config.get("model_paths")
    if model_paths and model_path in model_paths:
        resolved_path = model_paths[model_path]
        model["path"] = resolved_path
        logger.debug(f"Resolved model alias '{model_path}' -> '{resolved_path}'")

    # Resolve container alias
    container = model.get("container", "")

    containers = cluster_config.get("containers")
    if containers and container in containers:
        resolved_container = containers[container]
        model["container"] = resolved_container
        logger.debug(f"Resolved container alias '{container}' -> '{resolved_container}'")

    return config


def get_srtslurm_setting(key: str, default: Any = None) -> Any:
    """
    Get a setting from srtslurm.yaml cluster config.

    Args:
        key: Setting key (e.g., 'gpus_per_node', 'network_interface')
        default: Default value if not found

    Returns:
        Setting value or default if not found
    """
    cluster_config = load_cluster_config()
    if cluster_config and key in cluster_config:
        return cluster_config[key]
    return default


def load_config(
    path: Path | str,
    model_overrides: dict[str, Any] | None = None,
    slurm_overrides: dict[str, Any] | None = None,
) -> SrtConfig:
    """
    Load and validate YAML config, applying cluster defaults and CLI overrides.

    Priority order (highest to lowest):
    1. CLI overrides (model_overrides, slurm_overrides)
    2. YAML config file values
    3. Cluster defaults from srtslurm.yaml

    Returns a fully typed, frozen SrtConfig dataclass ready for use.

    Args:
        path: Path to the YAML configuration file
        model_overrides: Optional model overrides dict (e.g., {"path": "...", "container": "...", "precision": "..."})
        slurm_overrides: Optional SLURM overrides dict (e.g., {"account": "...", "partition": "...", "time_limit": "..."})

    Returns:
        SrtConfig frozen dataclass

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    # Load raw user config
    with open(path) as f:
        user_config = yaml.safe_load(f)

    # Apply CLI overrides first (highest priority)
    if model_overrides or slurm_overrides:
        user_config = merge_cli_overrides(user_config, model_overrides, slurm_overrides)

    # Load cluster defaults (optional)
    cluster_config = load_cluster_config()

    # Resolve with defaults (applies aliases and default values)
    resolved_config = resolve_config_with_defaults(user_config, cluster_config)

    # Parse with marshmallow schema to get typed SrtConfig
    try:
        schema = SrtConfig.Schema()
        config = schema.load(resolved_config)
        assert isinstance(config, SrtConfig)
        logger.info(f"Loaded config: {config.name}")
        return config
    except Exception as e:
        raise ValueError(f"Invalid config in {path}: {e}") from e

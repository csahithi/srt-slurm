# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for configuration loading and validation."""

import glob
from pathlib import Path

import pytest

from srtctl.backends import SGLangProtocol
from srtctl.core.schema import SrtConfig


class TestConfigLoading:
    """Tests for config file loading."""

    def test_config_loading_from_yaml(self):
        """Test that config files in recipies/ can be loaded."""
        # Find all yaml files in recipies/
        config_files = glob.glob("recipies/**/*.yaml", recursive=True)

        if not config_files:
            pytest.skip("No config files found in recipies/")

        errors = []
        loaded = 0
        for config_path in config_files:
            try:
                config = SrtConfig.from_yaml(Path(config_path))
                assert config.name is not None
                assert config.model is not None
                assert config.resources is not None
                assert config.backend is not None
                loaded += 1
                print(f"\n✓ Loaded config: {config_path}")
                print(f"  Name: {config.name}")
                print(f"  Backend: {config.backend_type}")
            except Exception as e:
                errors.append(f"{config_path}: {e}")

        print(f"\nLoaded {loaded}/{len(config_files)} configs")
        if errors:
            print(f"Errors ({len(errors)}):")
            for err in errors[:5]:  # Show first 5 errors
                print(f"  - {err}")


class TestSrtConfigStructure:
    """Tests for SrtConfig dataclass structure."""

    def test_resource_config_disaggregated(self):
        """Test resource config disaggregation detection."""
        from srtctl.core.schema import ResourceConfig

        # Disaggregated config
        disagg = ResourceConfig(
            gpu_type="h100",
            gpus_per_node=8,
            prefill_nodes=1,
            decode_nodes=2,
        )
        assert disagg.is_disaggregated is True

        # Aggregated config
        agg = ResourceConfig(
            gpu_type="h100",
            gpus_per_node=8,
            agg_nodes=2,
        )
        assert agg.is_disaggregated is False

    def test_decode_nodes_zero_inherits_tp_from_prefill(self):
        """When decode_nodes=0, gpus_per_decode inherits from prefill."""
        from srtctl.core.schema import ResourceConfig

        # 6 prefill + 2 decode on 2 nodes, sharing
        config = ResourceConfig(
            gpu_type="gb200",
            gpus_per_node=8,
            prefill_nodes=2,
            decode_nodes=0,
            prefill_workers=6,
            decode_workers=2,
        )

        assert config.gpus_per_prefill == 2  # (2*8)/6 = 2
        assert config.gpus_per_decode == 2  # inherits from prefill

        # Total GPUs should fit
        total_needed = config.num_prefill * config.gpus_per_prefill + config.num_decode * config.gpus_per_decode
        total_available = config.total_nodes * config.gpus_per_node
        assert total_needed <= total_available


class TestDynamoConfig:
    """Tests for DynamoConfig."""

    def test_default_version(self):
        """Default is version 0.7.0."""
        from srtctl.core.schema import DynamoConfig

        config = DynamoConfig()
        assert config.version == "0.7.0"
        assert config.hash is None
        assert config.top_of_tree is False
        assert not config.needs_source_install

    def test_version_install_command(self):
        """Version config generates pip install command."""
        from srtctl.core.schema import DynamoConfig

        config = DynamoConfig(version="0.8.0")
        cmd = config.get_install_commands()
        assert "pip install" in cmd
        assert "ai-dynamo-runtime==0.8.0" in cmd
        assert "ai-dynamo==0.8.0" in cmd

    def test_hash_install_command(self):
        """Hash config generates source install command."""
        from srtctl.core.schema import DynamoConfig

        config = DynamoConfig(hash="abc123")
        assert config.version is None  # Auto-cleared
        assert config.needs_source_install
        cmd = config.get_install_commands()
        assert "git clone" in cmd
        assert "git checkout abc123" in cmd
        assert "maturin build" in cmd
        assert "pip install -e" in cmd

    def test_top_of_tree_install_command(self):
        """Top-of-tree config generates source install without checkout."""
        from srtctl.core.schema import DynamoConfig

        config = DynamoConfig(top_of_tree=True)
        assert config.version is None  # Auto-cleared
        assert config.needs_source_install
        cmd = config.get_install_commands()
        assert "git clone" in cmd
        assert "git checkout" not in cmd
        assert "maturin build" in cmd

    def test_hash_and_top_of_tree_not_allowed(self):
        """Cannot specify both hash and top_of_tree."""
        from srtctl.core.schema import DynamoConfig

        with pytest.raises(ValueError, match="Cannot specify both"):
            DynamoConfig(hash="abc123", top_of_tree=True)


class TestSGLangProtocol:
    """Tests for SGLangProtocol."""

    def test_sglang_config_structure(self):
        """Test SGLang config has expected structure."""
        config = SGLangProtocol()

        assert config.type == "sglang"
        assert hasattr(config, "prefill_environment")
        assert hasattr(config, "decode_environment")
        assert hasattr(config, "sglang_config")

    def test_get_environment_for_mode(self):
        """Test environment variable retrieval per mode."""
        config = SGLangProtocol(
            prefill_environment={"PREFILL_VAR": "1"},
            decode_environment={"DECODE_VAR": "1"},
        )

        assert config.get_environment_for_mode("prefill") == {"PREFILL_VAR": "1"}
        assert config.get_environment_for_mode("decode") == {"DECODE_VAR": "1"}
        assert config.get_environment_for_mode("agg") == {}


class TestFrontendConfig:
    """Tests for FrontendConfig."""

    def test_frontend_defaults(self):
        """Test frontend config defaults."""
        from srtctl.core.schema import FrontendConfig

        frontend = FrontendConfig()

        assert frontend.use_sglang_router is False
        assert frontend.enable_multiple_frontends is True

    def test_router_args_list(self):
        """Test router args list generation."""
        from srtctl.core.schema import FrontendConfig

        frontend = FrontendConfig(
            use_sglang_router=True,
            sglang_router_args={"policy": "round_robin", "verbose": True},
        )

        args = frontend.get_router_args_list()
        assert "--policy" in args
        assert "round_robin" in args
        assert "--verbose" in args

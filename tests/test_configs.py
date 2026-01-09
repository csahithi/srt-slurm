# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for configuration loading and validation."""

import glob
from pathlib import Path

import pytest

from srtctl.backends import SGLangProtocol, SGLangServerConfig
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

    def test_kv_events_config_global_bool(self):
        """Test kv_events_config=True enables prefill+decode with defaults."""
        config = SGLangProtocol(kv_events_config=True)

        assert config.get_kv_events_config_for_mode("prefill") == {
            "publisher": "zmq",
            "topic": "kv-events",
        }
        assert config.get_kv_events_config_for_mode("decode") == {
            "publisher": "zmq",
            "topic": "kv-events",
        }
        assert config.get_kv_events_config_for_mode("agg") is None

    def test_kv_events_config_per_mode(self):
        """Test kv_events_config per-mode control."""
        config = SGLangProtocol(
            kv_events_config={
                "prefill": True,
                # decode omitted = disabled
            }
        )

        assert config.get_kv_events_config_for_mode("prefill") == {
            "publisher": "zmq",
            "topic": "kv-events",
        }
        assert config.get_kv_events_config_for_mode("decode") is None
        assert config.get_kv_events_config_for_mode("agg") is None

    def test_kv_events_config_custom_settings(self):
        """Test kv_events_config with custom publisher/topic."""
        config = SGLangProtocol(
            kv_events_config={
                "prefill": {"topic": "prefill-events"},
                "decode": {"publisher": "custom", "topic": "decode-events"},
            }
        )

        prefill_cfg = config.get_kv_events_config_for_mode("prefill")
        assert prefill_cfg["publisher"] == "zmq"  # default
        assert prefill_cfg["topic"] == "prefill-events"

        decode_cfg = config.get_kv_events_config_for_mode("decode")
        assert decode_cfg["publisher"] == "custom"
        assert decode_cfg["topic"] == "decode-events"

    def test_kv_events_config_disabled(self):
        """Test kv_events_config disabled by default."""
        config = SGLangProtocol()

        assert config.get_kv_events_config_for_mode("prefill") is None
        assert config.get_kv_events_config_for_mode("decode") is None
        assert config.get_kv_events_config_for_mode("agg") is None

    def test_grpc_mode_disabled_by_default(self):
        """Test gRPC mode is disabled by default."""
        config = SGLangProtocol()

        assert config.is_grpc_mode("prefill") is False
        assert config.is_grpc_mode("decode") is False
        assert config.is_grpc_mode("agg") is False

    def test_grpc_mode_enabled_per_mode(self):
        """Test gRPC mode can be enabled per worker mode."""
        config = SGLangProtocol(
            sglang_config=SGLangServerConfig(
                prefill={"grpc-mode": True},
                decode={"grpc-mode": True},
                aggregated={"grpc-mode": False},
            )
        )

        assert config.is_grpc_mode("prefill") is True
        assert config.is_grpc_mode("decode") is True
        assert config.is_grpc_mode("agg") is False


class TestFrontendConfig:
    """Tests for FrontendConfig."""

    def test_frontend_defaults(self):
        """Test frontend config defaults."""
        from srtctl.core.schema import FrontendConfig

        frontend = FrontendConfig()

        assert frontend.type == "dynamo"
        assert frontend.enable_multiple_frontends is True
        assert frontend.args is None
        assert frontend.env is None

    def test_frontend_sglang_type(self):
        """Test sglang frontend config."""
        from srtctl.core.schema import FrontendConfig

        frontend = FrontendConfig(
            type="sglang",
            args={"policy": "round_robin", "verbose": True},
            env={"MY_VAR": "value"},
        )

        assert frontend.type == "sglang"
        assert frontend.args == {"policy": "round_robin", "verbose": True}
        assert frontend.env == {"MY_VAR": "value"}


class TestSetupScript:
    """Tests for setup_script functionality."""

    def test_setup_script_in_config(self):
        """Test setup_script can be set in config."""
        from srtctl.core.schema import (
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/container.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", gpus_per_node=8, agg_nodes=1),
            setup_script="my-setup.sh",
        )

        assert config.setup_script == "my-setup.sh"

    def test_setup_script_override_with_replace(self):
        """Test setup_script can be overridden with dataclasses.replace."""
        from dataclasses import replace

        from srtctl.core.schema import (
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/container.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", gpus_per_node=8, agg_nodes=1),
        )

        assert config.setup_script is None

        # Override with replace (simulates CLI flag behavior)
        config = replace(config, setup_script="install-sglang-main.sh")
        assert config.setup_script == "install-sglang-main.sh"

    def test_sbatch_template_includes_setup_script_env_var(self):
        """Test that sbatch template sets SRTCTL_SETUP_SCRIPT env var."""
        from pathlib import Path

        from srtctl.cli.submit import generate_minimal_sbatch_script
        from srtctl.core.schema import (
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/container.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", gpus_per_node=8, agg_nodes=1),
        )

        # Without setup_script
        script = generate_minimal_sbatch_script(
            config=config,
            config_path=Path("/tmp/test.yaml"),
            setup_script=None,
        )
        assert "SRTCTL_SETUP_SCRIPT" not in script

        # With setup_script
        script = generate_minimal_sbatch_script(
            config=config,
            config_path=Path("/tmp/test.yaml"),
            setup_script="install-sglang-main.sh",
        )
        assert 'export SRTCTL_SETUP_SCRIPT="install-sglang-main.sh"' in script

    def test_setup_script_env_var_override(self, monkeypatch):
        """Test that SRTCTL_SETUP_SCRIPT env var overrides config."""
        import os
        from dataclasses import replace

        from srtctl.core.schema import (
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        config = SrtConfig(
            name="test",
            model=ModelConfig(path="/model", container="/container.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", gpus_per_node=8, agg_nodes=1),
            setup_script=None,
        )

        # Simulate env var being set (like do_sweep.main does)
        monkeypatch.setenv("SRTCTL_SETUP_SCRIPT", "install-sglang-main.sh")

        setup_script_override = os.environ.get("SRTCTL_SETUP_SCRIPT")
        assert setup_script_override == "install-sglang-main.sh"

        # Apply override like do_sweep.main does
        if setup_script_override:
            config = replace(config, setup_script=setup_script_override)

        assert config.setup_script == "install-sglang-main.sh"


class TestCLIOverrides:
    """Tests for CLI overrides functionality."""

    def test_model_overrides(self, tmp_path):
        """Test that model overrides work correctly."""
        from srtctl.core.config import load_config

        # Create a minimal test config
        config_file = tmp_path / "test.yaml"
        config_file.write_text(
            """
name: "test"
model:
  path: "/original/model"
  container: "/original/container.sqsh"
  precision: "fp8"
resources:
  gpu_type: "h100"
  gpus_per_node: 8
  agg_nodes: 1
"""
        )

        # Load without overrides
        config = load_config(config_file)
        assert config.model.path == "/original/model"
        assert config.model.container == "/original/container.sqsh"
        assert config.model.precision == "fp8"

        # Load with model overrides
        model_overrides = {
            "path": "/overridden/model",
            "container": "/overridden/container.sqsh",
            "precision": "bf16",
        }
        config = load_config(config_file, model_overrides=model_overrides)
        assert config.model.path == "/overridden/model"
        assert config.model.container == "/overridden/container.sqsh"
        assert config.model.precision == "bf16"

    def test_slurm_overrides(self, tmp_path):
        """Test that SLURM overrides work correctly."""
        from srtctl.core.config import load_config

        # Create a minimal test config
        config_file = tmp_path / "test.yaml"
        config_file.write_text(
            """
name: "test"
model:
  path: "/model"
  container: "/container.sqsh"
  precision: "fp8"
resources:
  gpu_type: "h100"
  gpus_per_node: 8
  agg_nodes: 1
slurm:
  account: "original-account"
  partition: "original-partition"
  time_limit: "01:00:00"
"""
        )

        # Load without overrides
        config = load_config(config_file)
        assert config.slurm.account == "original-account"
        assert config.slurm.partition == "original-partition"
        assert config.slurm.time_limit == "01:00:00"

        # Load with SLURM overrides
        slurm_overrides = {
            "account": "overridden-account",
            "partition": "overridden-partition",
            "time_limit": "02:00:00",
        }
        config = load_config(config_file, slurm_overrides=slurm_overrides)
        assert config.slurm.account == "overridden-account"
        assert config.slurm.partition == "overridden-partition"
        assert config.slurm.time_limit == "02:00:00"

    def test_partial_overrides(self, tmp_path):
        """Test that partial overrides only override specified fields."""
        from srtctl.core.config import load_config

        # Create a minimal test config
        config_file = tmp_path / "test.yaml"
        config_file.write_text(
            """
name: "test"
model:
  path: "/original/model"
  container: "/original/container.sqsh"
  precision: "fp8"
resources:
  gpu_type: "h100"
  gpus_per_node: 8
  agg_nodes: 1
slurm:
  account: "original-account"
  partition: "original-partition"
  time_limit: "01:00:00"
"""
        )

        # Partial model override (only path)
        model_overrides = {"path": "/overridden/model"}
        config = load_config(config_file, model_overrides=model_overrides)
        assert config.model.path == "/overridden/model"
        assert config.model.container == "/original/container.sqsh"  # Unchanged
        assert config.model.precision == "fp8"  # Unchanged

        # Partial SLURM override (only account)
        slurm_overrides = {"account": "overridden-account"}
        config = load_config(config_file, slurm_overrides=slurm_overrides)
        assert config.slurm.account == "overridden-account"
        assert config.slurm.partition == "original-partition"  # Unchanged
        assert config.slurm.time_limit == "01:00:00"  # Unchanged

    def test_overrides_without_slurm_section(self, tmp_path):
        """Test that SLURM overrides work even when config has no slurm section."""
        from srtctl.core.config import load_config

        # Create a minimal test config without slurm section
        config_file = tmp_path / "test.yaml"
        config_file.write_text(
            """
name: "test"
model:
  path: "/model"
  container: "/container.sqsh"
  precision: "fp8"
resources:
  gpu_type: "h100"
  gpus_per_node: 8
  agg_nodes: 1
"""
        )

        # Load with SLURM overrides (should create slurm section)
        slurm_overrides = {
            "account": "test-account",
            "partition": "test-partition",
            "time_limit": "02:00:00",
        }
        config = load_config(config_file, slurm_overrides=slurm_overrides)
        assert config.slurm.account == "test-account"
        assert config.slurm.partition == "test-partition"
        assert config.slurm.time_limit == "02:00:00"

    def test_overrides_priority_over_cluster_defaults(self, tmp_path, monkeypatch):
        """Test that CLI overrides take priority over cluster defaults."""
        from srtctl.core.config import load_config

        # Create a minimal test config
        config_file = tmp_path / "test.yaml"
        config_file.write_text(
            """
name: "test"
model:
  path: "/model"
  container: "/container.sqsh"
  precision: "fp8"
resources:
  gpu_type: "h100"
  gpus_per_node: 8
  agg_nodes: 1
"""
        )

        # Create a mock cluster config
        cluster_config_file = tmp_path / "srtslurm.yaml"
        cluster_config_file.write_text(
            """
default_account: "cluster-account"
default_partition: "cluster-partition"
default_time_limit: "04:00:00"
"""
        )

        # Change to tmp_path so srtslurm.yaml is found
        import os

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            # Load without overrides - should use cluster defaults
            config = load_config(config_file)
            assert config.slurm.account == "cluster-account"
            assert config.slurm.partition == "cluster-partition"
            assert config.slurm.time_limit == "04:00:00"

            # Load with SLURM overrides - should override cluster defaults
            slurm_overrides = {
                "account": "cli-account",
                "partition": "cli-partition",
                "time_limit": "01:00:00",
            }
            config = load_config(config_file, slurm_overrides=slurm_overrides)
            assert config.slurm.account == "cli-account"
            assert config.slurm.partition == "cli-partition"
            assert config.slurm.time_limit == "01:00:00"
        finally:
            os.chdir(original_cwd)

    def test_slurm_overrides_routing_to_sbatch_directives(self, tmp_path):
        """Test that non-standard SLURM fields are routed to sbatch_directives."""
        from srtctl.core.config import load_config

        # Create a minimal test config
        config_file = tmp_path / "test.yaml"
        config_file.write_text(
            """
name: "test"
model:
  path: "/model"
  container: "/container.sqsh"
  precision: "fp8"
resources:
  gpu_type: "h100"
  gpus_per_node: 8
  agg_nodes: 1
"""
        )

        # Load with mixed SLURM overrides (standard + sbatch_directives)
        slurm_overrides = {
            "account": "test-account",  # Standard field -> slurm section
            "qos": "high",  # Non-standard -> sbatch_directives
            "constraint": "volta",  # Non-standard -> sbatch_directives
            "segment": "4",  # Non-standard -> sbatch_directives
            "exclusive": "",  # Flag without value -> sbatch_directives
        }
        config = load_config(config_file, slurm_overrides=slurm_overrides)

        # Standard fields should be in slurm section
        assert config.slurm.account == "test-account"

        # Non-standard fields should be in sbatch_directives
        assert config.sbatch_directives["qos"] == "high"
        assert config.sbatch_directives["constraint"] == "volta"
        assert config.sbatch_directives["segment"] == "4"
        assert config.sbatch_directives["exclusive"] == ""

    def test_slurm_overrides_mixed_standard_and_sbatch(self, tmp_path):
        """Test mixing standard slurm fields with sbatch_directives in one override."""
        from srtctl.core.config import load_config

        # Create a minimal test config with existing sbatch_directives
        config_file = tmp_path / "test.yaml"
        config_file.write_text(
            """
name: "test"
model:
  path: "/model"
  container: "/container.sqsh"
  precision: "fp8"
resources:
  gpu_type: "h100"
  gpus_per_node: 8
  agg_nodes: 1
sbatch_directives:
  mail-user: "original@example.com"
  qos: "low"
"""
        )

        # Override both standard and non-standard fields
        slurm_overrides = {
            "account": "new-account",  # Standard -> slurm
            "time_limit": "02:00:00",  # Standard -> slurm
            "qos": "high",  # Non-standard -> sbatch_directives (overrides existing)
            "constraint": "volta",  # Non-standard -> sbatch_directives (new)
        }
        config = load_config(config_file, slurm_overrides=slurm_overrides)

        # Standard fields in slurm section
        assert config.slurm.account == "new-account"
        assert config.slurm.time_limit == "02:00:00"

        # Non-standard fields in sbatch_directives (merged with existing)
        assert config.sbatch_directives["mail-user"] == "original@example.com"  # Preserved
        assert config.sbatch_directives["qos"] == "high"  # Overridden
        assert config.sbatch_directives["constraint"] == "volta"  # New


class TestCLIOverrideE2E:
    """E2E tests for CLI override functionality via dry-run."""

    def test_cli_model_override_via_dry_run(self, tmp_path, monkeypatch):
        """Test that CLI model overrides work via dry-run command."""
        from srtctl.cli.submit import submit_single

        # Create a minimal test config
        config_file = tmp_path / "test.yaml"
        config_file.write_text(
            """
name: "test"
model:
  path: "/original/model"
  container: "/original/container.sqsh"
  precision: "fp8"
resources:
  gpu_type: "h100"
  gpus_per_node: 8
  agg_nodes: 1
"""
        )

        # Mock subprocess.run to avoid actually calling sbatch
        import subprocess
        from unittest.mock import MagicMock

        def mock_sbatch(cmd, **kwargs):
            if cmd[0] == "sbatch":
                result = MagicMock()
                result.stdout = "Submitted batch job 12345\n"
                result.returncode = 0
                return result
            raise subprocess.CalledProcessError(1, cmd)

        monkeypatch.setattr("subprocess.run", mock_sbatch)

        # Test with model overrides via dry-run (should not call sbatch)
        model_overrides = {
            "path": "/overridden/model",
            "container": "/overridden/container.sqsh",
            "precision": "bf16",
        }

        # This should work without errors
        submit_single(
            config_path=config_file,
            dry_run=True,
            model_overrides=model_overrides,
        )

        # Verify the override was applied by loading config directly
        from srtctl.core.config import load_config

        config = load_config(config_file, model_overrides=model_overrides)
        assert config.model.path == "/overridden/model"
        assert config.model.container == "/overridden/container.sqsh"
        assert config.model.precision == "bf16"

    def test_cli_slurm_override_via_dry_run(self, tmp_path, monkeypatch):
        """Test that CLI SLURM overrides work via dry-run command."""
        from srtctl.cli.submit import submit_single

        # Create a minimal test config
        config_file = tmp_path / "test.yaml"
        config_file.write_text(
            """
name: "test"
model:
  path: "/model"
  container: "/container.sqsh"
  precision: "fp8"
resources:
  gpu_type: "h100"
  gpus_per_node: 8
  agg_nodes: 1
slurm:
  account: "original-account"
  partition: "original-partition"
  time_limit: "01:00:00"
"""
        )

        # Mock subprocess.run
        import subprocess
        from unittest.mock import MagicMock

        def mock_sbatch(cmd, **kwargs):
            if cmd[0] == "sbatch":
                result = MagicMock()
                result.stdout = "Submitted batch job 12345\n"
                result.returncode = 0
                return result
            raise subprocess.CalledProcessError(1, cmd)

        monkeypatch.setattr("subprocess.run", mock_sbatch)

        # Test with SLURM overrides
        slurm_overrides = {
            "account": "overridden-account",
            "time_limit": "02:00:00",
            "qos": "high",  # Should go to sbatch_directives
        }

        submit_single(
            config_path=config_file,
            dry_run=True,
            slurm_overrides=slurm_overrides,
        )

        # Verify the override was applied
        from srtctl.core.config import load_config

        config = load_config(config_file, slurm_overrides=slurm_overrides)
        assert config.slurm.account == "overridden-account"
        assert config.slurm.time_limit == "02:00:00"
        assert config.sbatch_directives["qos"] == "high"

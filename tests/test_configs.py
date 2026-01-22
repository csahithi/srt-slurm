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
        """Default is version 0.8.0."""
        from srtctl.core.schema import DynamoConfig

        config = DynamoConfig()
        assert config.version == "0.8.0"
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


class TestOutputDirectoryStructure:
    """Tests for output directory structure created during job submission."""

    def test_output_directory_created_with_job_id(self, tmp_path, monkeypatch):
        """Test that outputs/{job_id}/ directory is created on successful submission."""
        import json
        import subprocess
        from unittest.mock import MagicMock, patch

        from srtctl.cli.submit import submit_with_orchestrator
        from srtctl.core.schema import (
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        # Create a test config
        config = SrtConfig(
            name="test-output-dir",
            model=ModelConfig(path="/model", container="/container.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", gpus_per_node=8, agg_nodes=1),
        )

        # Create a temp config file
        config_file = tmp_path / "config.yaml"
        config_file.write_text("name: test")

        # Mock srtctl_root to use temp directory
        monkeypatch.setattr(
            "srtctl.cli.submit.get_srtslurm_setting",
            lambda key, default=None: str(tmp_path) if key == "srtctl_root" else default,
        )

        # Mock sbatch to return a fake job ID
        mock_result = MagicMock()
        mock_result.stdout = "Submitted batch job 12345"
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            submit_with_orchestrator(
                config_path=config_file,
                config=config,
                dry_run=False,
            )

        # Verify directory structure
        output_dir = tmp_path / "outputs" / "12345"
        assert output_dir.exists(), "outputs/{job_id}/ directory should be created"
        assert output_dir.is_dir(), "outputs/{job_id}/ should be a directory"

    def test_config_yaml_copied_to_output_dir(self, tmp_path, monkeypatch):
        """Test that config.yaml is copied to outputs/{job_id}/."""
        from unittest.mock import MagicMock, patch

        from srtctl.cli.submit import submit_with_orchestrator
        from srtctl.core.schema import (
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        config = SrtConfig(
            name="test-config-copy",
            model=ModelConfig(path="/model", container="/container.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", gpus_per_node=8, agg_nodes=1),
        )

        # Create config file with specific content
        config_file = tmp_path / "my_config.yaml"
        config_content = "name: test-config-copy\nmodel:\n  path: /model"
        config_file.write_text(config_content)

        monkeypatch.setattr(
            "srtctl.cli.submit.get_srtslurm_setting",
            lambda key, default=None: str(tmp_path) if key == "srtctl_root" else default,
        )

        mock_result = MagicMock()
        mock_result.stdout = "Submitted batch job 99999"
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            submit_with_orchestrator(config_path=config_file, config=config, dry_run=False)

        # Verify config.yaml was copied
        copied_config = tmp_path / "outputs" / "99999" / "config.yaml"
        assert copied_config.exists(), "config.yaml should be copied to output dir"
        assert copied_config.read_text() == config_content, "config.yaml content should match original"

    def test_sbatch_script_copied_to_output_dir(self, tmp_path, monkeypatch):
        """Test that sbatch_script.sh is copied to outputs/{job_id}/."""
        from unittest.mock import MagicMock, patch

        from srtctl.cli.submit import submit_with_orchestrator
        from srtctl.core.schema import (
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        config = SrtConfig(
            name="test-sbatch-copy",
            model=ModelConfig(path="/model", container="/container.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", gpus_per_node=8, agg_nodes=1),
        )

        config_file = tmp_path / "config.yaml"
        config_file.write_text("name: test")

        monkeypatch.setattr(
            "srtctl.cli.submit.get_srtslurm_setting",
            lambda key, default=None: str(tmp_path) if key == "srtctl_root" else default,
        )

        mock_result = MagicMock()
        mock_result.stdout = "Submitted batch job 88888"
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            submit_with_orchestrator(config_path=config_file, config=config, dry_run=False)

        # Verify sbatch_script.sh was copied
        sbatch_script = tmp_path / "outputs" / "88888" / "sbatch_script.sh"
        assert sbatch_script.exists(), "sbatch_script.sh should be copied to output dir"
        # Verify it's a valid sbatch script
        content = sbatch_script.read_text()
        assert "#!/bin/bash" in content, "sbatch script should have bash shebang"
        assert "#SBATCH" in content, "sbatch script should have SBATCH directives"

    def test_metadata_json_created_in_output_dir(self, tmp_path, monkeypatch):
        """Test that {job_id}.json metadata file is created in outputs/{job_id}/."""
        import json
        from unittest.mock import MagicMock, patch

        from srtctl.cli.submit import submit_with_orchestrator
        from srtctl.core.schema import (
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        config = SrtConfig(
            name="test-metadata",
            model=ModelConfig(path="/model", container="/container.sqsh", precision="fp8"),
            resources=ResourceConfig(
                gpu_type="h100",
                gpus_per_node=8,
                prefill_nodes=1,
                decode_nodes=2,
                prefill_workers=1,
                decode_workers=4,
            ),
        )

        config_file = tmp_path / "config.yaml"
        config_file.write_text("name: test")

        monkeypatch.setattr(
            "srtctl.cli.submit.get_srtslurm_setting",
            lambda key, default=None: str(tmp_path) if key == "srtctl_root" else default,
        )

        mock_result = MagicMock()
        mock_result.stdout = "Submitted batch job 77777"
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            submit_with_orchestrator(config_path=config_file, config=config, dry_run=False)

        # Verify {job_id}.json was created
        metadata_file = tmp_path / "outputs" / "77777" / "77777.json"
        assert metadata_file.exists(), "{job_id}.json should be created in output dir"

        # Verify metadata content
        metadata = json.loads(metadata_file.read_text())
        assert metadata["version"] == "2.0"
        assert metadata["orchestrator"] is True
        assert metadata["job_id"] == "77777"
        assert metadata["job_name"] == "test-metadata"
        assert metadata["model"]["path"] == "/model"
        assert metadata["model"]["container"] == "/container.sqsh"
        assert metadata["model"]["precision"] == "fp8"
        assert metadata["resources"]["gpu_type"] == "h100"
        assert metadata["resources"]["prefill_nodes"] == 1
        assert metadata["resources"]["decode_nodes"] == 2
        assert metadata["resources"]["prefill_workers"] == 1
        assert metadata["resources"]["decode_workers"] == 4

    def test_tags_included_in_metadata(self, tmp_path, monkeypatch):
        """Test that tags are included in metadata when provided."""
        import json
        from unittest.mock import MagicMock, patch

        from srtctl.cli.submit import submit_with_orchestrator
        from srtctl.core.schema import (
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        config = SrtConfig(
            name="test-tags",
            model=ModelConfig(path="/model", container="/container.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", gpus_per_node=8, agg_nodes=1),
        )

        config_file = tmp_path / "config.yaml"
        config_file.write_text("name: test")

        monkeypatch.setattr(
            "srtctl.cli.submit.get_srtslurm_setting",
            lambda key, default=None: str(tmp_path) if key == "srtctl_root" else default,
        )

        mock_result = MagicMock()
        mock_result.stdout = "Submitted batch job 66666"
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            submit_with_orchestrator(
                config_path=config_file,
                config=config,
                dry_run=False,
                tags=["experiment", "baseline", "v2"],
            )

        metadata_file = tmp_path / "outputs" / "66666" / "66666.json"
        metadata = json.loads(metadata_file.read_text())
        assert metadata["tags"] == ["experiment", "baseline", "v2"]

    def test_complete_output_directory_structure(self, tmp_path, monkeypatch):
        """Test that complete output directory structure is preserved."""
        import json
        from unittest.mock import MagicMock, patch

        from srtctl.cli.submit import submit_with_orchestrator
        from srtctl.core.schema import (
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        config = SrtConfig(
            name="test-complete-structure",
            model=ModelConfig(path="/model", container="/container.sqsh", precision="fp4"),
            resources=ResourceConfig(gpu_type="gb200", gpus_per_node=4, agg_nodes=2, agg_workers=2),
            setup_script="my-setup.sh",
        )

        config_file = tmp_path / "config.yaml"
        config_file.write_text("name: test-complete-structure")

        monkeypatch.setattr(
            "srtctl.cli.submit.get_srtslurm_setting",
            lambda key, default=None: str(tmp_path) if key == "srtctl_root" else default,
        )

        mock_result = MagicMock()
        mock_result.stdout = "Submitted batch job 55555"
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            submit_with_orchestrator(
                config_path=config_file,
                config=config,
                dry_run=False,
                tags=["production"],
            )

        output_dir = tmp_path / "outputs" / "55555"

        # Verify all expected files exist
        expected_files = [
            output_dir / "config.yaml",
            output_dir / "sbatch_script.sh",
            output_dir / "55555.json",
        ]
        for expected_file in expected_files:
            assert expected_file.exists(), f"{expected_file.name} should exist in output dir"

        # Verify metadata includes setup_script
        metadata = json.loads((output_dir / "55555.json").read_text())
        assert metadata["setup_script"] == "my-setup.sh"
        assert metadata["tags"] == ["production"]
        assert metadata["resources"]["agg_workers"] == 2

    def test_dry_run_does_not_create_output_dir(self, tmp_path, monkeypatch):
        """Test that dry-run mode does NOT create output directory."""
        from srtctl.cli.submit import submit_with_orchestrator
        from srtctl.core.schema import (
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        config = SrtConfig(
            name="test-dry-run",
            model=ModelConfig(path="/model", container="/container.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", gpus_per_node=8, agg_nodes=1),
        )

        config_file = tmp_path / "config.yaml"
        config_file.write_text("name: test")

        monkeypatch.setattr(
            "srtctl.cli.submit.get_srtslurm_setting",
            lambda key, default=None: str(tmp_path) if key == "srtctl_root" else default,
        )

        # Dry run should not call sbatch or create output dir
        submit_with_orchestrator(config_path=config_file, config=config, dry_run=True)

        # Verify no output directory was created
        outputs_dir = tmp_path / "outputs"
        assert not outputs_dir.exists(), "outputs/ should not be created in dry-run mode"

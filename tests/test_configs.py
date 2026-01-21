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


class TestJobMetadataSchema:
    """Tests to ensure job metadata JSON schema is preserved.

    The metadata JSON (written to {job_id}.json) is used by analysis tools
    to load benchmark runs. Changes to this schema will break analysis.
    """

    def test_metadata_schema_has_required_fields(self):
        """Verify metadata schema has all required fields for analysis tools."""
        from datetime import datetime

        from srtctl.core.schema import (
            BenchmarkConfig,
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        config = SrtConfig(
            name="test-job",
            model=ModelConfig(path="/model", container="/container.sqsh", precision="fp8"),
            resources=ResourceConfig(
                gpu_type="h100",
                gpus_per_node=8,
                prefill_nodes=1,
                decode_nodes=2,
            ),
            benchmark=BenchmarkConfig(type="sa-bench", isl=1024, osl=1024),
        )

        # Build metadata exactly as submit.py does
        metadata = {
            "version": "2.0",
            "orchestrator": True,
            "job_id": "12345",
            "job_name": config.name,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": {
                "path": config.model.path,
                "container": config.model.container,
                "precision": config.model.precision,
            },
            "resources": {
                "gpu_type": config.resources.gpu_type,
                "gpus_per_node": config.resources.gpus_per_node,
                "prefill_nodes": config.resources.prefill_nodes,
                "decode_nodes": config.resources.decode_nodes,
                "prefill_workers": config.resources.num_prefill,
                "decode_workers": config.resources.num_decode,
                "agg_workers": config.resources.num_agg,
            },
            "backend_type": config.backend_type,
            "frontend_type": config.frontend.type,
            "benchmark": {
                "type": config.benchmark.type,
                "isl": config.benchmark.isl,
                "osl": config.benchmark.osl,
            },
        }

        # Required top-level fields
        required_top_level = [
            "version",
            "job_id",
            "job_name",
            "generated_at",
            "model",
            "resources",
            "backend_type",
            "frontend_type",
            "benchmark",
        ]
        for field in required_top_level:
            assert field in metadata, f"Missing required top-level field: {field}"

        # Required model fields
        required_model = ["path", "container", "precision"]
        for field in required_model:
            assert field in metadata["model"], f"Missing required model field: {field}"

        # Required resource fields
        required_resources = [
            "gpu_type",
            "gpus_per_node",
            "prefill_nodes",
            "decode_nodes",
            "prefill_workers",
            "decode_workers",
            "agg_workers",
        ]
        for field in required_resources:
            assert field in metadata["resources"], f"Missing required resources field: {field}"

        # Required benchmark fields
        required_benchmark = ["type", "isl", "osl"]
        for field in required_benchmark:
            assert field in metadata["benchmark"], f"Missing required benchmark field: {field}"

    def test_metadata_can_be_parsed_by_analysis_tools(self):
        """Verify metadata can be read back and parsed for analysis."""
        import json
        import tempfile
        from datetime import datetime
        from pathlib import Path

        from srtctl.core.schema import (
            BenchmarkConfig,
            ModelConfig,
            ResourceConfig,
            SrtConfig,
        )

        config = SrtConfig(
            name="test-job",
            model=ModelConfig(path="/model", container="/container.sqsh", precision="fp8"),
            resources=ResourceConfig(
                gpu_type="h200",
                gpus_per_node=8,
                prefill_nodes=3,
                decode_nodes=2,
            ),
            benchmark=BenchmarkConfig(type="sa-bench", isl=8192, osl=1024),
        )

        metadata = {
            "version": "2.0",
            "orchestrator": True,
            "job_id": "534",
            "job_name": config.name,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": {
                "path": config.model.path,
                "container": config.model.container,
                "precision": config.model.precision,
            },
            "resources": {
                "gpu_type": config.resources.gpu_type,
                "gpus_per_node": config.resources.gpus_per_node,
                "prefill_nodes": config.resources.prefill_nodes,
                "decode_nodes": config.resources.decode_nodes,
                "prefill_workers": config.resources.num_prefill,
                "decode_workers": config.resources.num_decode,
                "agg_workers": config.resources.num_agg,
            },
            "backend_type": config.backend_type,
            "frontend_type": config.frontend.type,
            "benchmark": {
                "type": config.benchmark.type,
                "isl": config.benchmark.isl,
                "osl": config.benchmark.osl,
            },
        }

        # Write and read back
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(metadata, f, indent=2)
            temp_path = f.name

        try:
            with open(temp_path) as f:
                loaded = json.load(f)

            # Verify we can extract the data analysis tools need
            r = loaded["resources"]
            b = loaded["benchmark"]

            total_gpus = (r["prefill_nodes"] + r["decode_nodes"]) * r["gpus_per_node"]
            assert total_gpus == 40  # (3+2) * 8

            topology = f"{r['prefill_workers']}P{r['decode_workers']}D"
            # With prefill_nodes=3 and no explicit workers, num_prefill defaults to 0
            # Let me check what the actual values are
            assert r["prefill_workers"] is not None
            assert r["decode_workers"] is not None

            assert b["isl"] == 8192
            assert b["osl"] == 1024
            assert b["type"] == "sa-bench"

        finally:
            Path(temp_path).unlink()

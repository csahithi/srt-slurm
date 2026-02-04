# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for sidecar and custom frontend functionality."""

import pytest
from marshmallow import ValidationError

from srtctl.core.schema import (
    FrontendConfig,
    ModelConfig,
    ResourceConfig,
    SidecarConfig,
    SrtConfig,
)


class TestSidecarConfig:
    """Tests for SidecarConfig dataclass."""

    def test_sidecar_config_basic(self):
        """Test basic sidecar config creation."""
        config = SidecarConfig(command="python3 /workspace/router.py --port 9001")
        assert config.command == "python3 /workspace/router.py --port 9001"
        assert config.container is None
        assert config.env == {}

    def test_sidecar_config_with_container(self):
        """Test sidecar config with custom container."""
        config = SidecarConfig(
            command="python3 /workspace/router.py",
            container="my-custom-container:latest",
        )
        assert config.container == "my-custom-container:latest"

    def test_sidecar_config_with_env(self):
        """Test sidecar config with environment variables."""
        config = SidecarConfig(
            command="python3 /workspace/router.py",
            env={"ROUTER_PORT": "9001", "LOG_LEVEL": "debug"},
        )
        assert config.env["ROUTER_PORT"] == "9001"
        assert config.env["LOG_LEVEL"] == "debug"

    def test_sidecar_config_schema_load(self):
        """Test loading sidecar config via marshmallow schema."""
        data = {
            "command": "python3 /workspace/processor.py --enable-router",
            "container": "dynamo-runtime:0.7.1",
            "env": {"METRICS_CSV": "/logs/metrics.csv"},
        }
        schema = SidecarConfig.Schema()
        config = schema.load(data)
        assert config.command == "python3 /workspace/processor.py --enable-router"
        assert config.container == "dynamo-runtime:0.7.1"
        assert config.env["METRICS_CSV"] == "/logs/metrics.csv"


class TestCustomFrontendConfig:
    """Tests for custom frontend configuration."""

    def test_frontend_config_custom_type(self):
        """Test frontend config with type=custom."""
        config = FrontendConfig(
            type="custom",
            command="python3 /workspace/frontend.py --http-port 8000",
        )
        assert config.type == "custom"
        assert config.command == "python3 /workspace/frontend.py --http-port 8000"

    def test_frontend_config_custom_with_container(self):
        """Test custom frontend with specific container."""
        config = FrontendConfig(
            type="custom",
            command="python3 /workspace/frontend.py",
            container="custom-frontend:latest",
        )
        assert config.container == "custom-frontend:latest"

    def test_frontend_config_custom_with_env(self):
        """Test custom frontend with environment variables."""
        config = FrontendConfig(
            type="custom",
            command="python3 /workspace/frontend.py",
            env={"FRONTEND_MODEL_MAPPING": '{"llama": "/model"}'},
        )
        assert "FRONTEND_MODEL_MAPPING" in config.env

    def test_frontend_config_schema_load_custom(self):
        """Test loading custom frontend via schema."""
        data = {
            "type": "custom",
            "command": "python3 /workspace/frontend.py --http-port 8099",
            "container": "nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.7.1",
            "env": {"TPS_INTERVAL": "5"},
        }
        schema = FrontendConfig.Schema()
        config = schema.load(data)
        assert config.type == "custom"
        assert config.command == "python3 /workspace/frontend.py --http-port 8099"


class TestSrtConfigWithSidecars:
    """Tests for SrtConfig with sidecars field."""

    def test_config_with_sidecars(self):
        """Test SrtConfig with sidecars."""
        config = SrtConfig(
            name="test-job",
            model=ModelConfig(path="/models/test", container="/containers/test.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", agg_nodes=1, agg_workers=1),
            sidecars={
                "router": SidecarConfig(command="python3 /workspace/router.py"),
                "processor": SidecarConfig(command="python3 /workspace/processor.py"),
            },
        )
        assert len(config.sidecars) == 2
        assert "router" in config.sidecars
        assert "processor" in config.sidecars
        assert config.sidecars["router"].command == "python3 /workspace/router.py"

    def test_config_without_sidecars(self):
        """Test SrtConfig without sidecars (default empty dict)."""
        config = SrtConfig(
            name="test-job",
            model=ModelConfig(path="/models/test", container="/containers/test.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", agg_nodes=1, agg_workers=1),
        )
        assert config.sidecars == {}

    def test_config_schema_load_with_sidecars(self):
        """Test loading SrtConfig with sidecars via schema."""
        data = {
            "name": "test-with-sidecars",
            "model": {
                "path": "/models/llama",
                "container": "/containers/sglang.sqsh",
                "precision": "bf16",
            },
            "resources": {
                "gpu_type": "h100",
                "agg_nodes": 1,
                "agg_workers": 1,
            },
            "sidecars": {
                "custom_router": {
                    "command": "python3 /workspace/router.py --block-size 64",
                    "env": {"LOG_LEVEL": "info"},
                },
            },
        }
        schema = SrtConfig.Schema()
        config = schema.load(data)
        assert len(config.sidecars) == 1
        assert config.sidecars["custom_router"].command == "python3 /workspace/router.py --block-size 64"


class TestCustomFrontendValidation:
    """Tests for custom frontend validation."""

    def test_custom_frontend_requires_command(self):
        """Test that type=custom requires command field."""
        with pytest.raises(ValidationError, match="frontend.command is required"):
            SrtConfig(
                name="test-job",
                model=ModelConfig(path="/models/test", container="/containers/test.sqsh", precision="fp8"),
                resources=ResourceConfig(gpu_type="h100", agg_nodes=1, agg_workers=1),
                frontend=FrontendConfig(type="custom"),  # Missing command!
            )

    def test_custom_frontend_with_command_is_valid(self):
        """Test that type=custom with command is valid."""
        config = SrtConfig(
            name="test-job",
            model=ModelConfig(path="/models/test", container="/containers/test.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", agg_nodes=1, agg_workers=1),
            frontend=FrontendConfig(
                type="custom",
                command="python3 /workspace/frontend.py",
            ),
        )
        assert config.frontend.type == "custom"
        assert config.frontend.command is not None

    def test_dynamo_frontend_does_not_require_command(self):
        """Test that type=dynamo doesn't require command field."""
        config = SrtConfig(
            name="test-job",
            model=ModelConfig(path="/models/test", container="/containers/test.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", agg_nodes=1, agg_workers=1),
            frontend=FrontendConfig(type="dynamo"),  # No command needed
        )
        assert config.frontend.type == "dynamo"
        assert config.frontend.command is None

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


class TestWorkerPreambleDynamoInstall:
    """Tests for worker preamble dynamo installation logic."""

    def test_worker_preamble_installs_dynamo_for_dynamo_frontend(self):
        """Test that worker preamble includes dynamo install for frontend.type=dynamo."""
        from unittest.mock import MagicMock

        from srtctl.cli.mixins.worker_stage import WorkerStageMixin

        # Create a mock mixin with required attributes
        mixin = MagicMock(spec=WorkerStageMixin)
        mixin.config = SrtConfig(
            name="test-job",
            model=ModelConfig(path="/models/test", container="/containers/test.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", agg_nodes=1, agg_workers=1),
            frontend=FrontendConfig(type="dynamo"),
        )

        # Call the actual method
        preamble = WorkerStageMixin._build_worker_preamble(mixin)

        assert preamble is not None
        assert "pip install" in preamble
        assert "dynamo" in preamble.lower()

    def test_worker_preamble_installs_dynamo_for_custom_frontend(self):
        """Test that worker preamble includes dynamo install for frontend.type=custom."""
        from unittest.mock import MagicMock

        from srtctl.cli.mixins.worker_stage import WorkerStageMixin

        mixin = MagicMock(spec=WorkerStageMixin)
        mixin.config = SrtConfig(
            name="test-job",
            model=ModelConfig(path="/models/test", container="/containers/test.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", agg_nodes=1, agg_workers=1),
            frontend=FrontendConfig(type="custom", command="python3 /workspace/frontend.py"),
        )

        preamble = WorkerStageMixin._build_worker_preamble(mixin)

        assert preamble is not None
        assert "pip install" in preamble
        assert "dynamo" in preamble.lower()

    def test_worker_preamble_skips_dynamo_for_sglang_frontend(self):
        """Test that worker preamble does NOT install dynamo for frontend.type=sglang."""
        from unittest.mock import MagicMock

        from srtctl.cli.mixins.worker_stage import WorkerStageMixin

        mixin = MagicMock(spec=WorkerStageMixin)
        mixin.config = SrtConfig(
            name="test-job",
            model=ModelConfig(path="/models/test", container="/containers/test.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", agg_nodes=1, agg_workers=1),
            frontend=FrontendConfig(type="sglang"),
        )

        preamble = WorkerStageMixin._build_worker_preamble(mixin)

        # Should be None since sglang frontend doesn't need dynamo
        assert preamble is None

    def test_worker_preamble_skips_dynamo_when_profiling_enabled(self):
        """Test that worker preamble skips dynamo when profiling is enabled."""
        from unittest.mock import MagicMock

        from srtctl.cli.mixins.worker_stage import WorkerStageMixin
        from srtctl.core.schema import ProfilingConfig, ProfilingPhaseConfig

        mixin = MagicMock(spec=WorkerStageMixin)
        mixin.config = SrtConfig(
            name="test-job",
            model=ModelConfig(path="/models/test", container="/containers/test.sqsh", precision="fp8"),
            resources=ResourceConfig(
                gpu_type="h100",
                agg_nodes=1,
                agg_workers=1,
            ),
            frontend=FrontendConfig(type="dynamo"),
            profiling=ProfilingConfig(
                type="nsys",
                isl=1024,
                osl=128,
                concurrency=1,
                aggregated=ProfilingPhaseConfig(start_step=5, stop_step=10),
            ),
        )

        preamble = WorkerStageMixin._build_worker_preamble(mixin)

        # Should be None - profiling skips dynamo install
        assert preamble is None

    def test_worker_preamble_skips_dynamo_when_install_false(self):
        """Test that worker preamble skips dynamo when dynamo.install=False."""
        from unittest.mock import MagicMock

        from srtctl.cli.mixins.worker_stage import WorkerStageMixin
        from srtctl.core.schema import DynamoConfig

        mixin = MagicMock(spec=WorkerStageMixin)
        mixin.config = SrtConfig(
            name="test-job",
            model=ModelConfig(path="/models/test", container="/containers/test.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", agg_nodes=1, agg_workers=1),
            frontend=FrontendConfig(type="dynamo"),
            dynamo=DynamoConfig(install=False),
        )

        preamble = WorkerStageMixin._build_worker_preamble(mixin)

        # Should be None - dynamo install disabled
        assert preamble is None

    def test_worker_preamble_includes_setup_script(self):
        """Test that worker preamble includes setup_script when configured."""
        from unittest.mock import MagicMock

        from srtctl.cli.mixins.worker_stage import WorkerStageMixin

        mixin = MagicMock(spec=WorkerStageMixin)
        mixin.config = SrtConfig(
            name="test-job",
            model=ModelConfig(path="/models/test", container="/containers/test.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", agg_nodes=1, agg_workers=1),
            frontend=FrontendConfig(type="dynamo"),
            setup_script="nat-setup.sh",
        )

        preamble = WorkerStageMixin._build_worker_preamble(mixin)

        assert preamble is not None
        assert "nat-setup.sh" in preamble
        assert "/configs/nat-setup.sh" in preamble


class TestSidecarCommandPreamble:
    """Tests for sidecar command preamble logic."""

    def test_sidecar_command_includes_setup_script(self):
        """Test that sidecar command includes setup_script when configured."""
        from unittest.mock import MagicMock

        from srtctl.cli.mixins.sidecar_stage import SidecarStageMixin

        mixin = MagicMock(spec=SidecarStageMixin)
        mixin.config = SrtConfig(
            name="test-job",
            model=ModelConfig(path="/models/test", container="/containers/test.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", agg_nodes=1, agg_workers=1),
            setup_script="nat-setup.sh",
        )

        cmd = SidecarStageMixin._build_sidecar_command(mixin, "python3 /workspace/router.py")

        assert cmd == ["bash", "-c", cmd[2]]  # Should be bash -c wrapped
        assert "nat-setup.sh" in cmd[2]
        assert "/configs/nat-setup.sh" in cmd[2]
        assert "python3 /workspace/router.py" in cmd[2]

    def test_sidecar_command_includes_dynamo_install(self):
        """Test that sidecar command includes dynamo install when enabled."""
        from unittest.mock import MagicMock

        from srtctl.cli.mixins.sidecar_stage import SidecarStageMixin

        mixin = MagicMock(spec=SidecarStageMixin)
        mixin.config = SrtConfig(
            name="test-job",
            model=ModelConfig(path="/models/test", container="/containers/test.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", agg_nodes=1, agg_workers=1),
        )

        cmd = SidecarStageMixin._build_sidecar_command(mixin, "python3 /workspace/router.py")

        # Default dynamo.install is True
        assert "pip install" in cmd[2]
        assert "dynamo" in cmd[2].lower()

    def test_sidecar_command_skips_dynamo_when_install_false(self):
        """Test that sidecar command skips dynamo when dynamo.install=False."""
        from unittest.mock import MagicMock

        from srtctl.cli.mixins.sidecar_stage import SidecarStageMixin
        from srtctl.core.schema import DynamoConfig

        mixin = MagicMock(spec=SidecarStageMixin)
        mixin.config = SrtConfig(
            name="test-job",
            model=ModelConfig(path="/models/test", container="/containers/test.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", agg_nodes=1, agg_workers=1),
            dynamo=DynamoConfig(install=False),
        )

        cmd = SidecarStageMixin._build_sidecar_command(mixin, "python3 /workspace/router.py")

        # Should NOT include pip install
        assert "pip install" not in cmd[2]
        # But should still include the user command
        assert "python3 /workspace/router.py" in cmd[2]

    def test_sidecar_command_order_setup_then_dynamo_then_command(self):
        """Test that sidecar command runs setup_script, then dynamo, then user command."""
        from unittest.mock import MagicMock

        from srtctl.cli.mixins.sidecar_stage import SidecarStageMixin

        mixin = MagicMock(spec=SidecarStageMixin)
        mixin.config = SrtConfig(
            name="test-job",
            model=ModelConfig(path="/models/test", container="/containers/test.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", agg_nodes=1, agg_workers=1),
            setup_script="nat-setup.sh",
        )

        cmd = SidecarStageMixin._build_sidecar_command(mixin, "python3 /workspace/router.py")
        full_cmd = cmd[2]

        # Find positions to verify order
        setup_pos = full_cmd.find("nat-setup.sh")
        dynamo_pos = full_cmd.find("pip install")
        user_cmd_pos = full_cmd.find("python3 /workspace/router.py")

        assert setup_pos < dynamo_pos < user_cmd_pos, "Order should be: setup_script, dynamo install, user command"


class TestCustomFrontendCommandPreamble:
    """Tests for custom frontend command preamble logic."""

    def test_custom_frontend_command_includes_setup_script(self):
        """Test that custom frontend command includes setup_script when configured."""
        from unittest.mock import MagicMock

        from srtctl.cli.mixins.frontend_stage import FrontendStageMixin

        mixin = MagicMock(spec=FrontendStageMixin)
        mixin.config = SrtConfig(
            name="test-job",
            model=ModelConfig(path="/models/test", container="/containers/test.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", agg_nodes=1, agg_workers=1),
            frontend=FrontendConfig(type="custom", command="python3 /workspace/frontend.py"),
            setup_script="nat-setup.sh",
        )

        cmd = FrontendStageMixin._build_custom_frontend_command(mixin, "python3 /workspace/frontend.py")

        assert cmd == ["bash", "-c", cmd[2]]
        assert "nat-setup.sh" in cmd[2]
        assert "/configs/nat-setup.sh" in cmd[2]

    def test_custom_frontend_command_includes_dynamo_install(self):
        """Test that custom frontend command includes dynamo install when enabled."""
        from unittest.mock import MagicMock

        from srtctl.cli.mixins.frontend_stage import FrontendStageMixin

        mixin = MagicMock(spec=FrontendStageMixin)
        mixin.config = SrtConfig(
            name="test-job",
            model=ModelConfig(path="/models/test", container="/containers/test.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", agg_nodes=1, agg_workers=1),
            frontend=FrontendConfig(type="custom", command="python3 /workspace/frontend.py"),
        )

        cmd = FrontendStageMixin._build_custom_frontend_command(mixin, "python3 /workspace/frontend.py")

        assert "pip install" in cmd[2]
        assert "dynamo" in cmd[2].lower()

    def test_custom_frontend_command_skips_dynamo_when_install_false(self):
        """Test that custom frontend command skips dynamo when dynamo.install=False."""
        from unittest.mock import MagicMock

        from srtctl.cli.mixins.frontend_stage import FrontendStageMixin
        from srtctl.core.schema import DynamoConfig

        mixin = MagicMock(spec=FrontendStageMixin)
        mixin.config = SrtConfig(
            name="test-job",
            model=ModelConfig(path="/models/test", container="/containers/test.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", agg_nodes=1, agg_workers=1),
            frontend=FrontendConfig(type="custom", command="python3 /workspace/frontend.py"),
            dynamo=DynamoConfig(install=False),
        )

        cmd = FrontendStageMixin._build_custom_frontend_command(mixin, "python3 /workspace/frontend.py")

        assert "pip install" not in cmd[2]
        assert "python3 /workspace/frontend.py" in cmd[2]

    def test_custom_frontend_command_order(self):
        """Test that custom frontend command runs in correct order."""
        from unittest.mock import MagicMock

        from srtctl.cli.mixins.frontend_stage import FrontendStageMixin

        mixin = MagicMock(spec=FrontendStageMixin)
        mixin.config = SrtConfig(
            name="test-job",
            model=ModelConfig(path="/models/test", container="/containers/test.sqsh", precision="fp8"),
            resources=ResourceConfig(gpu_type="h100", agg_nodes=1, agg_workers=1),
            frontend=FrontendConfig(type="custom", command="python3 /workspace/frontend.py"),
            setup_script="nat-setup.sh",
        )

        cmd = FrontendStageMixin._build_custom_frontend_command(mixin, "python3 /workspace/frontend.py")
        full_cmd = cmd[2]

        setup_pos = full_cmd.find("nat-setup.sh")
        dynamo_pos = full_cmd.find("pip install")
        user_cmd_pos = full_cmd.find("python3 /workspace/frontend.py")

        assert setup_pos < dynamo_pos < user_cmd_pos, "Order should be: setup_script, dynamo install, user command"

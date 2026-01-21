# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for analysis models."""

import pytest
from pydantic import ValidationError

from analysis.srtlog.models import RunMetadata


class TestRunMetadataFromJsonV2:
    """Tests for RunMetadata.from_json() v2.0 format parsing."""

    def test_v2_disaggregated_mode_basic(self):
        """Parse v2.0 format with disaggregated mode."""
        json_data = {
            "version": "2.0",
            "orchestrator": True,
            "job_id": "12345",
            "job_name": "test-benchmark",
            "generated_at": "2025-01-20 10:30:00",
            "partition": "gpu-partition",
            "model": {
                "path": "/models/llama-70b",
                "container": "/containers/sglang.sqsh",
                "precision": "fp8",
            },
            "resources": {
                "gpu_type": "h100",
                "gpus_per_node": 8,
                "prefill_nodes": 2,
                "decode_nodes": 3,
                "prefill_workers": 2,
                "decode_workers": 6,
                "agg_nodes": 0,
                "agg_workers": 0,
            },
            "frontend_type": "sglang",
            "enable_multiple_frontends": True,
            "num_additional_frontends": 2,
        }

        metadata = RunMetadata.from_json(json_data, "/logs/12345_test")

        assert metadata.job_id == "12345"
        assert metadata.job_name == "test-benchmark"
        assert metadata.run_date == "2025-01-20 10:30:00"
        assert metadata.partition == "gpu-partition"
        assert metadata.path == "/logs/12345_test"

        # Model config
        assert metadata.model_dir == "/models/llama-70b"
        assert metadata.container == "/containers/sglang.sqsh"
        assert metadata.precision == "fp8"

        # Resource config
        assert metadata.gpu_type == "h100"
        assert metadata.gpus_per_node == 8
        assert metadata.prefill_nodes == 2
        assert metadata.decode_nodes == 3
        assert metadata.prefill_workers == 2
        assert metadata.decode_workers == 6
        assert metadata.agg_nodes == 0
        assert metadata.agg_workers == 0

        # Frontend config
        assert metadata.frontend_type == "sglang"
        assert metadata.enable_multiple_frontends is True
        assert metadata.num_additional_frontends == 2

    def test_v2_aggregated_mode(self):
        """Parse v2.0 format with aggregated mode."""
        json_data = {
            "version": "2.0",
            "job_id": 67890,  # Test int conversion
            "job_name": "agg-benchmark",
            "generated_at": "2025-01-20 11:00:00",
            "model": {
                "path": "/models/qwen-32b",
                "container": "/containers/trtllm.sqsh",
                "precision": "fp4",
            },
            "resources": {
                "gpu_type": "gb200",
                "gpus_per_node": 8,
                "prefill_nodes": 0,
                "decode_nodes": 0,
                "prefill_workers": 0,
                "decode_workers": 0,
                "agg_nodes": 4,
                "agg_workers": 8,
            },
        }

        metadata = RunMetadata.from_json(json_data, "/logs/67890_agg")

        assert metadata.job_id == "67890"  # Should be string
        assert metadata.agg_nodes == 4
        assert metadata.agg_workers == 8
        assert metadata.prefill_nodes == 0
        assert metadata.decode_nodes == 0

    def test_v2_detected_by_orchestrator_flag(self):
        """v2.0 format detected via orchestrator flag without version."""
        json_data = {
            "orchestrator": True,
            "job_id": "11111",
            "job_name": "orchestrator-job",
            "generated_at": "2025-01-20 12:00:00",
            "model": {
                "path": "/models/test",
                "container": "/test.sqsh",
                "precision": "bf16",
            },
            "resources": {
                "gpu_type": "a100",
                "gpus_per_node": 8,
                "prefill_nodes": 1,
                "decode_nodes": 1,
                "prefill_workers": 1,
                "decode_workers": 2,
            },
        }

        metadata = RunMetadata.from_json(json_data, "/logs/11111")

        assert metadata.job_id == "11111"
        assert metadata.precision == "bf16"

    def test_v2_default_values(self):
        """Missing optional fields get default values."""
        json_data = {
            "version": "2.0",
            "job_id": "22222",
            "job_name": "minimal-job",
            "generated_at": "2025-01-20 13:00:00",
            "model": {
                "path": "/models/test",
                "container": "/test.sqsh",
                "precision": "fp8",
            },
            "resources": {
                "gpu_type": "h100",
                "gpus_per_node": 8,
                "prefill_nodes": 1,
                "decode_nodes": 1,
                "prefill_workers": 1,
                "decode_workers": 1,
            },
            # Missing: partition, frontend_type, enable_multiple_frontends, num_additional_frontends
        }

        metadata = RunMetadata.from_json(json_data, "/logs/22222")

        assert metadata.partition == ""
        assert metadata.frontend_type == "dynamo"
        assert metadata.enable_multiple_frontends is False
        assert metadata.num_additional_frontends == 0
        assert metadata.agg_nodes == 0
        assert metadata.agg_workers == 0

    def test_v2_empty_resources_section(self):
        """Handle missing resources section gracefully."""
        json_data = {
            "version": "2.0",
            "job_id": "33333",
            "job_name": "no-resources",
            "generated_at": "2025-01-20 14:00:00",
            "model": {
                "path": "/models/test",
                "container": "/test.sqsh",
                "precision": "fp8",
            },
            # Missing resources section entirely
        }

        metadata = RunMetadata.from_json(json_data, "/logs/33333")

        assert metadata.gpus_per_node == 0
        assert metadata.prefill_nodes == 0
        assert metadata.decode_nodes == 0
        assert metadata.gpu_type == ""

    def test_v2_empty_model_section(self):
        """Handle missing model section gracefully."""
        json_data = {
            "version": "2.0",
            "job_id": "44444",
            "job_name": "no-model",
            "generated_at": "2025-01-20 15:00:00",
            "resources": {
                "gpu_type": "h100",
                "gpus_per_node": 8,
                "agg_nodes": 1,
                "agg_workers": 1,
            },
            # Missing model section entirely
        }

        metadata = RunMetadata.from_json(json_data, "/logs/44444")

        assert metadata.model_dir == ""
        assert metadata.container == ""
        assert metadata.precision == ""

    def test_v1_format_raises_error(self):
        """v1.0 format (legacy) raises ValueError."""
        json_data = {
            "run_metadata": {
                "slurm_job_id": "55555",
                "job_name": "legacy-job",
            }
        }

        with pytest.raises(ValueError, match="Unsupported version"):
            RunMetadata.from_json(json_data, "/logs/55555")

    def test_mode_property_disaggregated(self):
        """mode property returns 'disaggregated' when prefill/decode nodes set."""
        json_data = {
            "version": "2.0",
            "job_id": "66666",
            "job_name": "disagg",
            "generated_at": "2025-01-20",
            "model": {"path": "/m", "container": "/c", "precision": "fp8"},
            "resources": {
                "gpu_type": "h100",
                "gpus_per_node": 8,
                "prefill_nodes": 2,
                "decode_nodes": 3,
                "prefill_workers": 2,
                "decode_workers": 3,
            },
        }

        metadata = RunMetadata.from_json(json_data, "/logs/66666")
        assert metadata.mode == "disaggregated"

    def test_mode_property_aggregated(self):
        """mode property returns 'aggregated' when agg_nodes set."""
        json_data = {
            "version": "2.0",
            "job_id": "77777",
            "job_name": "agg",
            "generated_at": "2025-01-20",
            "model": {"path": "/m", "container": "/c", "precision": "fp8"},
            "resources": {
                "gpu_type": "h100",
                "gpus_per_node": 8,
                "agg_nodes": 2,
                "agg_workers": 4,
            },
        }

        metadata = RunMetadata.from_json(json_data, "/logs/77777")
        assert metadata.mode == "aggregated"


class TestRunMetadataValidation:
    """Tests for RunMetadata pydantic validation."""

    def test_validation_rejects_mixed_mode(self):
        """Cannot have both disagg and agg fields set."""
        json_data = {
            "version": "2.0",
            "job_id": "88888",
            "job_name": "mixed-invalid",
            "generated_at": "2025-01-20",
            "model": {"path": "/m", "container": "/c", "precision": "fp8"},
            "resources": {
                "gpu_type": "h100",
                "gpus_per_node": 8,
                "prefill_nodes": 1,
                "decode_nodes": 2,
                "prefill_workers": 1,
                "decode_workers": 2,
                "agg_nodes": 1,  # Invalid: mixing modes
                "agg_workers": 1,
            },
        }

        with pytest.raises(ValidationError):
            RunMetadata.from_json(json_data, "/logs/88888")

    def test_validation_allows_zero_agg_with_disagg(self):
        """Zero agg values are allowed with disagg mode."""
        json_data = {
            "version": "2.0",
            "job_id": "99999",
            "job_name": "disagg-with-zeros",
            "generated_at": "2025-01-20",
            "model": {"path": "/m", "container": "/c", "precision": "fp8"},
            "resources": {
                "gpu_type": "h100",
                "gpus_per_node": 8,
                "prefill_nodes": 1,
                "decode_nodes": 2,
                "prefill_workers": 1,
                "decode_workers": 2,
                "agg_nodes": 0,  # Zero is fine
                "agg_workers": 0,
            },
        }

        # Should not raise
        metadata = RunMetadata.from_json(json_data, "/logs/99999")
        assert metadata.agg_nodes == 0


class TestRunMetadataProperties:
    """Tests for RunMetadata computed properties."""

    def test_is_aggregated_property(self):
        """is_aggregated property works correctly."""
        json_data = {
            "version": "2.0",
            "job_id": "10001",
            "job_name": "agg-test",
            "generated_at": "2025-01-20",
            "model": {"path": "/m", "container": "/c", "precision": "fp8"},
            "resources": {
                "gpu_type": "h100",
                "gpus_per_node": 8,
                "agg_nodes": 2,
                "agg_workers": 4,
            },
        }

        metadata = RunMetadata.from_json(json_data, "/logs/10001")
        assert metadata.is_aggregated is True

    def test_total_gpus_disaggregated(self):
        """total_gpus computed for disaggregated mode."""
        json_data = {
            "version": "2.0",
            "job_id": "10002",
            "job_name": "gpu-count",
            "generated_at": "2025-01-20",
            "model": {"path": "/m", "container": "/c", "precision": "fp8"},
            "resources": {
                "gpu_type": "h100",
                "gpus_per_node": 8,
                "prefill_nodes": 2,
                "decode_nodes": 3,
                "prefill_workers": 2,
                "decode_workers": 3,
            },
        }

        metadata = RunMetadata.from_json(json_data, "/logs/10002")
        # (2 + 3) nodes * 8 gpus = 40
        assert metadata.total_gpus == 40

    def test_total_gpus_aggregated(self):
        """total_gpus computed for aggregated mode."""
        json_data = {
            "version": "2.0",
            "job_id": "10003",
            "job_name": "gpu-count-agg",
            "generated_at": "2025-01-20",
            "model": {"path": "/m", "container": "/c", "precision": "fp8"},
            "resources": {
                "gpu_type": "h100",
                "gpus_per_node": 8,
                "agg_nodes": 4,
                "agg_workers": 8,
            },
        }

        metadata = RunMetadata.from_json(json_data, "/logs/10003")
        # 4 nodes * 8 gpus = 32
        assert metadata.total_gpus == 32


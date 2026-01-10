# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for AI-powered post-processing and failure analysis."""

import os
from unittest.mock import MagicMock, patch

import pytest

from srtctl.core.schema import AIAnalysisConfig, DEFAULT_AI_ANALYSIS_PROMPT


class TestAIAnalysisConfig:
    """Tests for AIAnalysisConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AIAnalysisConfig()

        assert config.enabled is False
        assert config.anthropic_api_key is None
        assert config.gh_token is None
        assert config.repos_to_search == ["sgl-project/sglang", "ai-dynamo/dynamo"]
        assert config.pr_search_days == 14
        assert config.prompt is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AIAnalysisConfig(
            enabled=True,
            anthropic_api_key="sk-test-key",
            gh_token="ghp_test_token",
            repos_to_search=["my-org/my-repo"],
            pr_search_days=7,
            prompt="Custom prompt: {log_dir}",
        )

        assert config.enabled is True
        assert config.anthropic_api_key == "sk-test-key"
        assert config.gh_token == "ghp_test_token"
        assert config.repos_to_search == ["my-org/my-repo"]
        assert config.pr_search_days == 7
        assert config.prompt == "Custom prompt: {log_dir}"

    def test_get_prompt_with_default(self):
        """Test get_prompt uses default template when prompt is None."""
        config = AIAnalysisConfig()
        prompt = config.get_prompt("/path/to/logs")

        assert "/path/to/logs" in prompt
        assert "sgl-project/sglang, ai-dynamo/dynamo" in prompt
        assert "14" in prompt  # pr_search_days

    def test_get_prompt_with_custom_template(self):
        """Test get_prompt uses custom template."""
        config = AIAnalysisConfig(
            prompt="Analyze logs in {log_dir}, search {repos} for last {pr_days} days",
            repos_to_search=["my-repo"],
            pr_search_days=7,
        )
        prompt = config.get_prompt("/my/logs")

        assert prompt == "Analyze logs in /my/logs, search my-repo for last 7 days"

    def test_get_prompt_variable_substitution(self):
        """Test all template variables are substituted."""
        config = AIAnalysisConfig(
            repos_to_search=["repo1", "repo2", "repo3"],
            pr_search_days=30,
        )
        prompt = config.get_prompt("/test/dir")

        assert "/test/dir" in prompt
        assert "repo1, repo2, repo3" in prompt
        assert "30" in prompt


class TestDefaultPrompt:
    """Tests for the default AI analysis prompt."""

    def test_default_prompt_has_placeholders(self):
        """Test default prompt has all required placeholders."""
        assert "{log_dir}" in DEFAULT_AI_ANALYSIS_PROMPT
        assert "{repos}" in DEFAULT_AI_ANALYSIS_PROMPT
        assert "{pr_days}" in DEFAULT_AI_ANALYSIS_PROMPT

    def test_default_prompt_mentions_gh_cli(self):
        """Test default prompt tells Claude about gh CLI."""
        assert "gh" in DEFAULT_AI_ANALYSIS_PROMPT.lower()
        assert "github" in DEFAULT_AI_ANALYSIS_PROMPT.lower() or "PR" in DEFAULT_AI_ANALYSIS_PROMPT

    def test_default_prompt_mentions_output_file(self):
        """Test default prompt tells Claude to write ai_analysis.md."""
        assert "ai_analysis.md" in DEFAULT_AI_ANALYSIS_PROMPT


class TestClusterConfigIntegration:
    """Tests for AI analysis config in cluster config."""

    def test_cluster_config_with_ai_analysis(self):
        """Test ClusterConfig can include AIAnalysisConfig."""
        from srtctl.core.schema import ClusterConfig

        cluster_config = ClusterConfig(
            default_account="test-account",
            ai_analysis=AIAnalysisConfig(
                enabled=True,
                anthropic_api_key="sk-test",
            ),
        )

        assert cluster_config.ai_analysis is not None
        assert cluster_config.ai_analysis.enabled is True
        assert cluster_config.ai_analysis.anthropic_api_key == "sk-test"

    def test_cluster_config_without_ai_analysis(self):
        """Test ClusterConfig works without AIAnalysisConfig."""
        from srtctl.core.schema import ClusterConfig

        cluster_config = ClusterConfig(
            default_account="test-account",
        )

        assert cluster_config.ai_analysis is None


class TestPostProcessStageMixin:
    """Tests for PostProcessStageMixin."""

    def test_resolve_secret_from_config(self):
        """Test secret resolution prefers config value."""
        from srtctl.cli.mixins.postprocess_stage import PostProcessStageMixin

        mixin = PostProcessStageMixin()
        result = mixin._resolve_secret("config-value", "ENV_VAR")

        assert result == "config-value"

    def test_resolve_secret_from_env(self, monkeypatch):
        """Test secret resolution falls back to environment variable."""
        from srtctl.cli.mixins.postprocess_stage import PostProcessStageMixin

        monkeypatch.setenv("TEST_SECRET", "env-value")

        mixin = PostProcessStageMixin()
        result = mixin._resolve_secret(None, "TEST_SECRET")

        assert result == "env-value"

    def test_resolve_secret_returns_none_when_not_found(self, monkeypatch):
        """Test secret resolution returns None when not found anywhere."""
        from srtctl.cli.mixins.postprocess_stage import PostProcessStageMixin

        # Ensure env var is not set
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)

        mixin = PostProcessStageMixin()
        result = mixin._resolve_secret(None, "NONEXISTENT_VAR")

        assert result is None

    def test_run_postprocess_skips_on_success(self):
        """Test run_postprocess does nothing when exit_code is 0."""
        from srtctl.cli.mixins.postprocess_stage import PostProcessStageMixin

        mixin = PostProcessStageMixin()
        mixin._get_ai_analysis_config = MagicMock()

        # Should not call _get_ai_analysis_config when exit_code is 0
        mixin.run_postprocess(0)

        mixin._get_ai_analysis_config.assert_not_called()

    def test_run_postprocess_skips_when_not_configured(self):
        """Test run_postprocess does nothing when AI analysis is not configured."""
        from srtctl.cli.mixins.postprocess_stage import PostProcessStageMixin

        mixin = PostProcessStageMixin()
        mixin._get_ai_analysis_config = MagicMock(return_value=None)
        mixin._run_ai_analysis = MagicMock()

        mixin.run_postprocess(1)

        mixin._get_ai_analysis_config.assert_called_once()
        mixin._run_ai_analysis.assert_not_called()

    def test_run_postprocess_skips_when_disabled(self):
        """Test run_postprocess does nothing when AI analysis is disabled."""
        from srtctl.cli.mixins.postprocess_stage import PostProcessStageMixin

        mixin = PostProcessStageMixin()
        mixin._get_ai_analysis_config = MagicMock(
            return_value=AIAnalysisConfig(enabled=False)
        )
        mixin._run_ai_analysis = MagicMock()

        mixin.run_postprocess(1)

        mixin._get_ai_analysis_config.assert_called_once()
        mixin._run_ai_analysis.assert_not_called()

    def test_run_postprocess_calls_ai_analysis_when_enabled(self):
        """Test run_postprocess calls _run_ai_analysis when enabled."""
        from srtctl.cli.mixins.postprocess_stage import PostProcessStageMixin

        config = AIAnalysisConfig(enabled=True, anthropic_api_key="sk-test")

        mixin = PostProcessStageMixin()
        mixin._get_ai_analysis_config = MagicMock(return_value=config)
        mixin._run_ai_analysis = MagicMock()

        mixin.run_postprocess(1)

        mixin._run_ai_analysis.assert_called_once_with(config)


class TestAIAnalysisConfigSchema:
    """Tests for AIAnalysisConfig marshmallow schema."""

    def test_schema_load_minimal(self):
        """Test loading minimal config from dict."""
        schema = AIAnalysisConfig.Schema()
        config = schema.load({"enabled": True})

        assert config.enabled is True
        assert config.repos_to_search == ["sgl-project/sglang", "ai-dynamo/dynamo"]

    def test_schema_load_full(self):
        """Test loading full config from dict."""
        schema = AIAnalysisConfig.Schema()
        config = schema.load({
            "enabled": True,
            "anthropic_api_key": "sk-test",
            "gh_token": "ghp_test",
            "repos_to_search": ["my/repo"],
            "pr_search_days": 7,
            "prompt": "Custom prompt",
        })

        assert config.enabled is True
        assert config.anthropic_api_key == "sk-test"
        assert config.gh_token == "ghp_test"
        assert config.repos_to_search == ["my/repo"]
        assert config.pr_search_days == 7
        assert config.prompt == "Custom prompt"

    def test_schema_dump(self):
        """Test dumping config to dict."""
        config = AIAnalysisConfig(
            enabled=True,
            anthropic_api_key="sk-test",
        )
        schema = AIAnalysisConfig.Schema()
        data = schema.dump(config)

        assert data["enabled"] is True
        assert data["anthropic_api_key"] == "sk-test"
        assert data["pr_search_days"] == 14  # default

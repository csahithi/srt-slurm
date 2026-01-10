# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Post-process stage mixin for SweepOrchestrator.

Handles AI-powered failure analysis using Claude Code CLI in headless mode.
Uses OpenRouter for authentication (simple API key, works in headless environments).
See: https://openrouter.ai/docs/guides/guides/claude-code-integration
"""

import logging
import os
import shlex
import subprocess
import time
from typing import TYPE_CHECKING

from srtctl.core.config import load_cluster_config
from srtctl.core.schema import AIAnalysisConfig
from srtctl.core.slurm import start_srun_process

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig

logger = logging.getLogger(__name__)


class PostProcessStageMixin:
    """Mixin for post-process stage after benchmark completion.

    Handles AI-powered failure analysis using Claude Code CLI.
    Configuration is loaded from srtslurm.yaml (cluster config).

    Requires:
        self.config: SrtConfig
        self.runtime: RuntimeContext
    """

    # Type hints for mixin dependencies
    config: "SrtConfig"
    runtime: "RuntimeContext"

    def _get_ai_analysis_config(self) -> AIAnalysisConfig | None:
        """Load AI analysis config from cluster config.

        Returns:
            AIAnalysisConfig if configured and enabled, None otherwise
        """
        cluster_config = load_cluster_config()
        if not cluster_config:
            return None

        ai_config_dict = cluster_config.get("ai_analysis")
        if not ai_config_dict:
            return None

        # Parse with schema
        try:
            schema = AIAnalysisConfig.Schema()
            return schema.load(ai_config_dict)
        except Exception as e:
            logger.warning("Failed to parse ai_analysis config: %s", e)
            return None

    def _resolve_secret(self, config_value: str | None, env_var: str) -> str | None:
        """Resolve a secret from config or environment variable.

        Args:
            config_value: Value from config (may be None)
            env_var: Environment variable name to check as fallback

        Returns:
            Resolved secret value, or None if not found
        """
        if config_value:
            return config_value
        return os.environ.get(env_var)

    def run_postprocess(self, exit_code: int) -> None:
        """Run post-processing after benchmark completion.

        Currently handles AI-powered failure analysis when enabled.

        Args:
            exit_code: Exit code from the benchmark run
        """
        if exit_code == 0:
            logger.debug("Benchmark succeeded, skipping AI analysis")
            return

        ai_config = self._get_ai_analysis_config()
        if not ai_config:
            logger.debug("AI analysis not configured in srtslurm.yaml")
            return

        if not ai_config.enabled:
            logger.debug("AI analysis is disabled")
            return

        logger.info("Running AI-powered failure analysis...")
        self._run_ai_analysis(ai_config)

    def _run_ai_analysis(self, config: AIAnalysisConfig) -> None:
        """Run AI analysis using Claude Code CLI via OpenRouter.

        Uses OpenRouter for authentication which works well in headless environments.
        See: https://openrouter.ai/docs/guides/guides/claude-code-integration

        Args:
            config: AI analysis configuration
        """
        # Resolve secrets
        openrouter_key = self._resolve_secret(config.openrouter_api_key, "OPENROUTER_API_KEY")
        gh_token = self._resolve_secret(config.gh_token, "GH_TOKEN")

        if not openrouter_key:
            logger.error("AI analysis requires OPENROUTER_API_KEY (set in srtslurm.yaml or environment)")
            return

        if not gh_token:
            logger.warning("GH_TOKEN not set - GitHub PR search will not work")

        # Build the prompt
        log_dir = str(self.runtime.log_dir)
        prompt = config.get_prompt(log_dir)

        logger.info("Log directory: %s", log_dir)
        logger.info("Repos to search: %s", ", ".join(config.repos_to_search))

        # Build environment variables for OpenRouter integration
        # Per https://openrouter.ai/docs/guides/guides/claude-code-integration
        env_to_set = {
            "ANTHROPIC_BASE_URL": "https://openrouter.ai/api",
            "ANTHROPIC_AUTH_TOKEN": openrouter_key,
            "ANTHROPIC_API_KEY": "",  # Must be explicitly empty to prevent conflicts
        }
        if gh_token:
            env_to_set["GH_TOKEN"] = gh_token

        # Build the Claude Code command
        # Use -p for headless mode and --dangerously-skip-permissions for full tool access
        claude_cmd = [
            "claude",
            "-p",
            prompt,
            "--dangerously-skip-permissions",
        ]

        # Wrap in bash to cd to log directory first
        bash_cmd = f"cd {shlex.quote(log_dir)} && {shlex.join(claude_cmd)}"

        analysis_log = self.runtime.log_dir / "ai_analysis.log"
        logger.info("Starting Claude Code analysis (log: %s)", analysis_log)

        try:
            proc = start_srun_process(
                command=["bash", "-c", bash_cmd],
                nodelist=[self.runtime.nodes.head],
                output=str(analysis_log),
                container_image=str(self.runtime.container_image),
                container_mounts=self.runtime.container_mounts,
                env_to_set=env_to_set,
            )

            # Wait for completion with timeout (10 minutes max)
            timeout = 600
            start_time = time.time()

            while proc.poll() is None:
                if time.time() - start_time > timeout:
                    logger.warning("AI analysis timed out after %d seconds", timeout)
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    return
                time.sleep(5)

            exit_code = proc.returncode or 0

            if exit_code != 0:
                logger.warning("AI analysis exited with code %d", exit_code)
            else:
                logger.info("AI analysis completed successfully")

            # Check if analysis file was created
            analysis_file = self.runtime.log_dir / "ai_analysis.md"
            if analysis_file.exists():
                logger.info("Analysis report written to: %s", analysis_file)
            else:
                logger.warning("AI analysis did not produce ai_analysis.md")

        except Exception as e:
            logger.error("Failed to run AI analysis: %s", e)

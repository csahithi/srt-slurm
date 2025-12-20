# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Logging utilities for srtctl.

Provides consistent logging configuration, emoji constants, and helper functions
for formatted output throughout the codebase.
"""

import logging
import sys

# ============================================================================
# Emoji Constants
# ============================================================================

CHECK = "✓"
CROSS = "✗"
ROCKET = "🚀"
GEAR = "⚙"
HOURGLASS = "⏳"
PACKAGE = "📦"
WRENCH = "🔧"
WARN = "⚠"
INFO = "ℹ"
SPARKLE = "✨"
FIRE = "🔥"
CLOCK = "🕐"
FOLDER = "📁"
FILE = "📄"
LINK = "🔗"
SERVER = "🖥"
GPU = "🎮"


# ============================================================================
# Logging Configuration
# ============================================================================


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.

    Uses the srtctl root logger configuration.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def setup_logging(
    level: int = logging.INFO,
    format_string: str | None = None,
    date_format: str | None = None,
) -> None:
    """Configure logging for srtctl.

    Sets up the root logger with consistent formatting.

    Args:
        level: Logging level (default: INFO)
        format_string: Custom format string (default: timestamp + level + message)
        date_format: Custom date format (default: ISO-like)
    """
    if format_string is None:
        format_string = "%(asctime)s [%(levelname)s] %(message)s"

    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"

    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt=date_format,
        stream=sys.stdout,
        force=True,  # Override any existing configuration
    )

    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)


# ============================================================================
# Output Helpers
# ============================================================================


def section(title: str, emoji: str = GEAR, logger: logging.Logger | None = None) -> None:
    """Print a section header.

    Creates a visually distinct section break in the logs.

    Args:
        title: Section title
        emoji: Emoji to prefix (default: gear)
        logger: Logger to use (default: root logger)
    """
    if logger is None:
        logger = logging.getLogger()

    logger.info("")
    logger.info("%s %s", emoji, title)
    logger.info("-" * 60)


def success(message: str, logger: logging.Logger | None = None) -> None:
    """Log a success message with checkmark.

    Args:
        message: Success message
        logger: Logger to use (default: root logger)
    """
    if logger is None:
        logger = logging.getLogger()
    logger.info("%s %s", CHECK, message)


def error(message: str, logger: logging.Logger | None = None) -> None:
    """Log an error message with cross.

    Args:
        message: Error message
        logger: Logger to use (default: root logger)
    """
    if logger is None:
        logger = logging.getLogger()
    logger.error("%s %s", CROSS, message)


def warn(message: str, logger: logging.Logger | None = None) -> None:
    """Log a warning message with warning emoji.

    Args:
        message: Warning message
        logger: Logger to use (default: root logger)
    """
    if logger is None:
        logger = logging.getLogger()
    logger.warning("%s %s", WARN, message)


def step(message: str, logger: logging.Logger | None = None) -> None:
    """Log a step/progress message with rocket.

    Args:
        message: Step message
        logger: Logger to use (default: root logger)
    """
    if logger is None:
        logger = logging.getLogger()
    logger.info("%s %s", ROCKET, message)


def waiting(message: str, logger: logging.Logger | None = None) -> None:
    """Log a waiting/pending message with hourglass.

    Args:
        message: Waiting message
        logger: Logger to use (default: root logger)
    """
    if logger is None:
        logger = logging.getLogger()
    logger.info("%s %s", HOURGLASS, message)


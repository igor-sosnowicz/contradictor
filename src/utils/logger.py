"""Module with a logger."""

import sys

from loguru import logger

from src.configuration import config


def get_configured_logger() -> None:
    """Configure a project-wide logger."""
    logger.remove()
    logger.add(sys.stderr, level=config.log_level.upper())


get_configured_logger()

__all__ = ["logger"]

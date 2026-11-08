"""Shared types for component configuration values."""

from pathlib import Path

type ConfigurationValue = (
    str
    | float
    | int
    | bool
    | Path
    | None
    | tuple[str, ...]
    | dict[str, "ConfigurationValue"]
)

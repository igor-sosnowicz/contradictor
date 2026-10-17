"""
Unified Path System v2: the single source of filesystem truth (spec-v2.md).

Usage patterns and the full contract: `docs/UPS.md`.
"""

from src.paths.router import (
    cache_path,
    get_cache_path,
    get_data_path,
    get_tmp_path,
    initialise,
    model_path,
    paths,
    processed_dataset_path,
    raw_dataset_path,
)

__all__ = [
    "cache_path",
    "get_cache_path",
    "get_data_path",
    "get_tmp_path",
    "initialise",
    "model_path",
    "paths",
    "processed_dataset_path",
    "raw_dataset_path",
]

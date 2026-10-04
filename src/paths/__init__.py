"""Single truth source for all filesystem paths."""

from src.paths.registry import (
    cache_path,
    ensure_all,
    initialise,
    model_path,
    processed_dataset_path,
    raw_dataset_path,
    resolve,
)

__all__ = [
    "cache_path",
    "ensure_all",
    "initialise",
    "model_path",
    "processed_dataset_path",
    "raw_dataset_path",
    "resolve",
]

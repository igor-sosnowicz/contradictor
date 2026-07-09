"""Module with a dataset utilities useful in multiple modules."""

from pathlib import Path

from src.configuration import config


def to_raw_dataset_path(raw_dataset_name: str) -> Path:
    """
    Convert a sole name to path of a raw (unprocessed) dataset.

    This function does NOT guarantee the dataset will exist under this path.
    Only that it will be deterministic and conformat with the configuration.

    Args:
        raw_dataset_name (str): Name of the raw dataset.

    Returns:
        Path: Path where a raw dataset should be.
    """
    raw_dataset_name = raw_dataset_name.replace(" ", "_").replace("-", "_")
    return config.data_directory / config.raw_dataset_subdirectory / raw_dataset_name

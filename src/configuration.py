"""Module with a project-wide configuration object."""

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class Configuration(BaseModel):
    """The project-wide configuration."""

    # Forbid extra parameters absent from the configuration.
    model_config = ConfigDict(extra="forbid")

    data_directory: Path = Path("./data")

    # A sub-directory of the data directory storing raw versions of downloaded datasets.
    raw_dataset_subdirectory: str = "raw_datasets"
    preprocessed_dataset_subdirectory: str = "processed_dataset"

    # In seconds.
    dataset_download_timeout: int = 300


def load_configuration(file: Path) -> Configuration:
    """
    Load configuration from a configuration file.

    Args:
        file (Path): Path to a configuration file.

    Returns:
        Configuration: Configuration object read to use.
    """
    with file.open("rb") as f:
        config_data = tomllib.load(f)

    return Configuration(**config_data)


config = load_configuration(Path("./config.toml"))

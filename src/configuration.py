"""Module with a project-wide configuration object."""

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from src.argument_detection.config import (
    XGBoostExtractorConfig,
)


class Configuration(BaseModel):
    """The project-wide configuration."""

    # Forbid extra parameters absent from the configuration.
    model_config = ConfigDict(extra="forbid")

    cache_directory: Path = Path("./.cache")
    data_directory: Path = Path("./data")

    # A sub-directory of the data directory storing raw versions of downloaded datasets.
    raw_dataset_subdirectory: str = "raw_datasets"
    preprocessed_dataset_subdirectory: str = "processed_dataset"
    model_subdirectory: str = "models"

    xgboost_extractor: XGBoostExtractorConfig = Field(
        default_factory=XGBoostExtractorConfig,
    )

    # In seconds.
    dataset_download_timeout: int = 300

    style_vector_dimensions: int = Field(5, ge=1)


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


config: Configuration = load_configuration(Path("./config.toml"))

"""Module with a project-wide configuration object."""

import tempfile
import tomllib
from getpass import getuser
from pathlib import Path
from typing import Annotated, Literal

import pydantic
import torch
from platformdirs import PlatformDirs
from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from src.argument_detection.config import (
    XGBoostExtractorConfig,
)
from src.data_models.data_models import (
    ArgumentExtractorImplementation,
    ArgumentFramerImplementation,
    EncoderImplementation,
    NLIImplementation,
    SearchPipelineImplementation,
    StyleExtractorImplementation,
    VectorSearchImplementation,
)
from src.data_models.validators import validate_device_presence
from src.utils.errors import ConfigurationError

directories = PlatformDirs("Contradictor")


class Configuration(BaseModel):
    """The project-wide configuration."""

    # Forbid extra parameters absent from the configuration.
    model_config = ConfigDict(extra="forbid")

    # Default models.
    transformer_nli_model: str = "tasksource/ModernBERT-base-nli"

    # Cached data that can be regenerated.
    cache_directory: Path = directories.user_cache_path

    # Persistent application data.
    data_directory: Path = directories.user_data_path

    # Ephemeral application files that may be discarded by the OS between runs.
    tmp_directory: Path = Path(tempfile.gettempdir()) / f"Contradictor-{getuser()}"

    # A sub-directory of the data directory storing raw versions of downloaded datasets.
    raw_dataset_subdirectory: str = "raw_datasets"
    preprocessed_dataset_subdirectory: str = "processed_dataset"
    model_subdirectory: str = "models"

    xgboost_extractor: XGBoostExtractorConfig = XGBoostExtractorConfig()

    # In seconds.
    dataset_download_timeout: int = 300

    reference_text_max_length: int = Field(
        default=10_000,
        ge=1,
        description="A maximum number of characters a reference text can have.",
    )

    encoder_output_dimensions: int = Field(
        10,
        ge=1,
        description=(
            "A number of dimensions of a representation vector generated "
            "by the encoder."
        ),
    )

    # Contradictor pipeline implementations
    search_pipeline: SearchPipelineImplementation = SearchPipelineImplementation.REMOTE
    argument_extractor: ArgumentExtractorImplementation = (
        ArgumentExtractorImplementation.XGBOOST
    )
    argument_framer: ArgumentFramerImplementation = ArgumentFramerImplementation.XGBOOST
    nli: NLIImplementation = NLIImplementation.TRANSFORMERS_MODERNBERT
    style_extractor: StyleExtractorImplementation = StyleExtractorImplementation.SPACY
    encoder: EncoderImplementation = EncoderImplementation.MLP
    vector_search: VectorSearchImplementation = VectorSearchImplementation.NUMPY_BASED

    random_seed: int | None = Field(
        None,
        description=(
            "A seed for a random number generators. An integer ensures "
            "reproducibility while None initialises the seed randomly."
        ),
    )

    device: Annotated[str, AfterValidator(validate_device_presence)] = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Argument encoding dataset. Importance of semantics for distance.
    # Style weight is 1.0 - encoding_dataset_semantics_weight.
    encoding_dataset_semantics_weight: float = 0.7

    counterargument_candidates_limit: int = Field(
        100,
        description=(
            "A maximum number of counterargument candidates considered by the pipeline."
        ),
        gt=0,
    )

    log_level: Literal["debug", "info", "warning", "error", "critical"] = "debug"

    api_host: str = Field("127.0.0.1", min_length=3)
    api_port: int = Field(5555, ge=0, le=65_535)
    reference_text_max_characters: int = Field(
        50_000,
        gt=1,
    )


def load_configuration(file: Path | None) -> Configuration:
    """
    Load configuration from a configuration file or
    return default configuration if no file is provided.

    Args:
        file (Path | None): Path to a configuration file.

    Returns:
        Configuration: Configuration object read to use.

    Raises:
        ConfigurationError: Raised if there is a configuration error.
    """
    if file is None:
        log.info("No configuration file provided, using default configuration.")
        return Configuration()

    try:
        with file.open("rb") as f:
            config_data = tomllib.load(f)
        return Configuration(**config_data)
    except (ValueError, KeyError, pydantic.ValidationError) as e:
        raise ConfigurationError(str(e)) from e
    except FileNotFoundError as e:
        raise ConfigurationError(
            f"The configuration file is missing: {file.resolve()}"
        ) from e


def _resolve_configuration_file() -> Path | None:
    """
    Return path to config file to load, or None when none exists.

    Search order:
    1. Dev worktree (`./config.toml`)
    2. PlatformDirs (`~/.config/Contradictor/config.toml`)
    """
    candidates = (
        Path("./config.toml"),
        directories.user_config_path / "config.toml",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


config: Configuration = load_configuration(_resolve_configuration_file())

"""Module with a project-wide configuration object."""

import tomllib
from pathlib import Path

import pydantic
from pydantic import BaseModel, ConfigDict, Field

from src.argument_detection.config import (
    XGBoostExtractorConfig,
)
from src.data_models.data_models import (
    ArgumentExtractorImplementation,
    ArgumentFramerImplementation,
    ComputingBackend,
    EncoderImplementation,
    NLIImplementation,
    SearchPipelineImplementation,
    StyleExtractorImplementation,
    VectorSearchImplementation,
)
from src.utils.errors import ConfigurationError


class Configuration(BaseModel):
    """The project-wide configuration."""

    # Default models.
    transformer_nli_model: str = "tasksource/ModernBERT-base-nli"

    # Forbid extra parameters absent from the configuration.
    model_config = ConfigDict(extra="forbid")

    cache_directory: Path = Path("./.cache")
    data_directory: Path = Path("./data")

    # A sub-directory of the data directory storing raw versions of downloaded datasets.
    raw_dataset_subdirectory: str = "raw_datasets"
    preprocessed_dataset_subdirectory: str = "processed_dataset"
    model_subdirectory: str = "models"

    xgboost_extractor: XGBoostExtractorConfig = XGBoostExtractorConfig()

    # In seconds.
    dataset_download_timeout: int = 300

    computing_backend: ComputingBackend = ComputingBackend.CPU

    reference_text_max_length: int = Field(
        10_000,
        ge=1,
        description="A maximum number of characters a reference text can have.",
    )
    vector_search_max_candidates: int = Field(10, ge=1)

    # Contradictor pipeline implementations
    search_pipeline: SearchPipelineImplementation = (
        SearchPipelineImplementation.SELF_IMPLEMENTED
    )
    argument_extractor: ArgumentExtractorImplementation = (
        ArgumentExtractorImplementation.NOT_IMPLEMENTED
    )
    argument_framer: ArgumentFramerImplementation = ArgumentFramerImplementation.XGBOOST
    nli: NLIImplementation = NLIImplementation.TRANSFORMERS_MODERNBERT
    style_extractor: StyleExtractorImplementation = StyleExtractorImplementation.SPACY
    encoder: EncoderImplementation = EncoderImplementation.NOT_IMPLEMENTED
    vector_search: VectorSearchImplementation = (
        VectorSearchImplementation.NOT_IMPLEMENTED
    )


def load_configuration(file: Path) -> Configuration:
    """
    Load configuration from a configuration file.

    Args:
        file (Path): Path to a configuration file.

    Returns:
        Configuration: Configuration object read to use.
    """
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


config: Configuration = load_configuration(Path("./config.toml"))

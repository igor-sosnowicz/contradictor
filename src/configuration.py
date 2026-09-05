"""Module with a project-wide configuration object."""

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_DEFAULT_CACHE_DIR = Path("./.cache")
_DEFAULT_DATA_DIR = Path("./data")
_DEFAULT_SOURCES_DIR = Path("./data/counterargument-dataset/sources")
_DEFAULT_RECORDS_DIR = Path("./contr-argument-dataset/records")


class Configuration(BaseModel):
    """The project-wide configuration."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    # Storage directories
    cache_directory: Path = Field(
        default_factory=lambda: _DEFAULT_CACHE_DIR,
        description="Directory for caching temporary files",
    )
    data_directory: Path = Field(
        default_factory=lambda: _DEFAULT_DATA_DIR,
        description="Directory for storing data files",
    )
    counterargument_dataset_path: Path = Field(
        default_factory=lambda: (
            _DEFAULT_DATA_DIR / "counterargument-dataset/dataset.json"
        ),
        description="Path to the aggregated argument database JSON",
    )
    counterargument_dataset_embeddings_path: Path = Field(
        default_factory=lambda: (
            _DEFAULT_DATA_DIR / "counterargument-dataset/embeddings"
        ),
        description="Path to the aggregated argument embeddings pickle",
    )
    raw_dataset_subdirectory: Path = Field(
        default_factory=lambda: _DEFAULT_DATA_DIR / "raw_dataset"
    )
    preprocessed_dataset_subdirectory: Path = Field(
        default_factory=lambda: _DEFAULT_DATA_DIR / "processed_dataset"
    )
    model_subdirectory: Path = Field(
        default_factory=lambda: _DEFAULT_DATA_DIR / "models"
    )

    # contr-argument-dataset working directories (embedding-free intermediates).
    dataset_sources_directory: Path = Field(
        default_factory=lambda: _DEFAULT_SOURCES_DIR,
        description="Directory with raw source documents (*.txt/*.md)",
    )
    dataset_records_directory: Path = Field(
        default_factory=lambda: _DEFAULT_RECORDS_DIR,
        description="Directory with per-document records JSON (no embeddings)",
    )

    # In seconds.
    dataset_download_timeout: int = 300
    style_vector_dimensions: int = Field(5, ge=1)

    # LM Studio API configuration
    lm_studio_api_base_url: str = "localhost:1234"
    lm_studio_api_key: str = "your_api_key_here"

    # FastEmbed model configuration
    provider: Literal["api", "local"] = "local"
    sparse_model_name: str = "Qdrant/bm25"
    dense_model_name: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    fastembed_model_cache_directory: Path = Field(
        default_factory=lambda: _DEFAULT_CACHE_DIR / "fastembed_models",
        description="Cache directory for fastembed models",
    )

    # LLM pipeline (LM Studio) configuration. Model names are NOT hardcoded:
    # they must be provided via config.toml or environment to match the
    # models actually loaded in LM Studio.
    llm_backend: Literal["openai_compat", "lms_sdk"] = Field(
        default="openai_compat",
        description="Which client to use for LM Studio",
    )
    extractor_model: str | None = Field(
        default=None,
        description="Large model for extraction (loaded in LM Studio)",
    )
    judge_model: str | None = Field(
        default=None,
        description="Small model for judging (loaded in LM Studio)",
    )
    pipeline_chunk_size: int = Field(default=2000, ge=200)
    pipeline_chunk_overlap: int = Field(default=200, ge=0)
    pipeline_top_k_links: int = Field(default=5, ge=1)

    def _ensure_derived_paths(self) -> None:
        """Re-derive optional paths when base directories were customized."""
        data_custom = self.data_directory != _DEFAULT_DATA_DIR
        cache_custom = self.cache_directory != _DEFAULT_CACHE_DIR
        rebase: dict[str, tuple[Path, Path, Path]] = {
            # field, default base, default value
            "counterargument_dataset_path": (
                _DEFAULT_DATA_DIR,
                _DEFAULT_DATA_DIR / "counterargument-dataset/datasett.json",
                self.data_directory / "counterargument-dataset/datasett.json",
            ),
            "counterargument_dataset_embeddings_path": (
                _DEFAULT_DATA_DIR,
                _DEFAULT_DATA_DIR / "counterargument-dataset/embeddings",
                self.data_directory / "counterargument-dataset/embeddings",
            ),
            "raw_dataset_subdirectory": (
                _DEFAULT_DATA_DIR,
                _DEFAULT_DATA_DIR / "raw_dataset",
                self.data_directory / "raw_dataset",
            ),
            "preprocessed_dataset_subdirectory": (
                _DEFAULT_DATA_DIR,
                _DEFAULT_DATA_DIR / "processed_dataset",
                self.data_directory / "processed_dataset",
            ),
            "model_subdirectory": (
                _DEFAULT_DATA_DIR,
                _DEFAULT_DATA_DIR / "models",
                self.data_directory / "models",
            ),
            "fastembed_model_cache_directory": (
                _DEFAULT_CACHE_DIR,
                _DEFAULT_CACHE_DIR / "fastembed_models",
                self.cache_directory / "fastembed_models",
            ),
        }
        for field_name, (base, default_value, rebased_value) in rebase.items():
            customized = (data_custom and base == _DEFAULT_DATA_DIR) or (
                cache_custom and base == _DEFAULT_CACHE_DIR
            )
            if customized and getattr(self, field_name) == default_value:
                object.__setattr__(self, field_name, rebased_value)

    def _ensure_directories(self) -> None:
        """Create directories for all path settings."""
        for field_name in (
            "cache_directory",
            "data_directory",
            "raw_dataset_subdirectory",
            "preprocessed_dataset_subdirectory",
            "model_subdirectory",
            "dataset_sources_directory",
            "dataset_records_directory",
            "fastembed_model_cache_directory",
        ):
            field_value = getattr(self, field_name)
            if isinstance(field_value, Path):
                field_value.mkdir(parents=True, exist_ok=True)
        for file_parent in (
            self.counterargument_dataset_path,
            self.counterargument_dataset_embeddings_path,
        ):
            if isinstance(file_parent, Path):
                file_parent.parent.mkdir(parents=True, exist_ok=True)

    def model_post_init(self, __context: object, /) -> None:
        """Populate derived paths and create directories."""
        self._ensure_derived_paths()
        self._ensure_directories()


def _default_config_path() -> Path:
    """Resolve config.toml relative to the repository root, not CWD."""
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        candidate = parent / "config.toml"
        if candidate.exists():
            return candidate
    return Path("./config.toml")


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


def _load_global_config() -> Configuration:
    try:
        return load_configuration(_default_config_path())
    except FileNotFoundError:
        return Configuration.model_validate({})


config = _load_global_config()

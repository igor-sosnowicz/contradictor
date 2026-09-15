"""Module with a project-wide configuration object."""

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_DEFAULT_CACHE_DIR = Path("./.cache")
_DEFAULT_DATA_DIR = Path("./data")


class RoleModelSettings(BaseModel):
    """Sampling parameters for an agent role."""

    model_config = ConfigDict(extra="forbid")

    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    max_tokens: int | None = Field(
        default=None, ge=1, description="Overrides the shared llm.max_tokens"
    )
    seed: int | None = Field(
        default=None,
        description=("Fixed seed. Only for roles that must be reproducible."),
    )
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] | None = (
        Field(
            default=None,
            description=(
                "Sent as OpenAI's reasoning_effort. 'none' turns thinking off, "
                "which on a local reasoning model is a ~5x speedup: the reasoning "
                "block is the bulk of the output and none of it is the answer."
            ),
        )
    )


class LLMSettings(BaseModel):
    """Per-role sampling parameters shared by every agent."""

    model_config = ConfigDict(extra="forbid")

    timeout: float = Field(
        default=120.0,
        gt=0.0,
        description="Per-request timeout, stuck model fails instead of hanging",
    )
    max_tokens: int = Field(
        default=2048,
        ge=1,
        description=(
            "Shared output budget. Reasoning models need a high value: their "
            "reasoning tokens count towards it, and truncation returns garbage."
        ),
    )

    # Reproducible, near-greedym seeded.
    # Thinking is off by default: these roles fill in a small JSON schema, where
    # the reasoning block is pure overhead. Raise it per role in config.toml if
    # a task turns out to need deliberation.
    extractor: RoleModelSettings = Field(
        default_factory=lambda: RoleModelSettings(
            temperature=0.1, seed=0, reasoning_effort="none"
        )
    )
    judge: RoleModelSettings = Field(
        default_factory=lambda: RoleModelSettings(
            temperature=0.0, seed=0, reasoning_effort="none"
        )
    )

    # Creative
    generator: RoleModelSettings = Field(
        default_factory=lambda: RoleModelSettings(
            temperature=0.8, reasoning_effort="none"
        )
    )
    styler: RoleModelSettings = Field(
        default_factory=lambda: RoleModelSettings(
            temperature=0.7, reasoning_effort="none"
        )
    )


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
    raw_dataset_subdirectory: Path = Field(
        default_factory=lambda: _DEFAULT_DATA_DIR / "raw_dataset"
    )
    preprocessed_dataset_subdirectory: Path = Field(
        default_factory=lambda: _DEFAULT_DATA_DIR / "processed_dataset"
    )
    model_subdirectory: Path = Field(
        default_factory=lambda: _DEFAULT_DATA_DIR / "models"
    )

    # --- Golden dataset thresholds ---
    premise_contradiction_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "NLI contradiction probability above which generated premises are "
            "thrown away before they ever reach the support judge."
        ),
    )
    premise_support_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum support score the judge must give for generated premises "
            "to be kept. NOT an entailment score - a premise supports a claim "
            "without logically entailing it."
        ),
    )
    premise_synthesis_retries: int = Field(
        default=2,
        ge=0,
        description="Regeneration attempts before leaving a record without premises",
    )
    nli_model_name: str = Field(
        default="cross-encoder/nli-deberta-v3-base",
        description="Cross-encoder used to score premise -> conclusion entailment",
    )
    dataset_download_timeout: int = 300  # seconds
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
    llm: LLMSettings = Field(default_factory=LLMSettings)

    def _ensure_derived_paths(self) -> None:
        """Re-derive optional paths when base directories were customized."""
        data_custom = self.data_directory != _DEFAULT_DATA_DIR
        cache_custom = self.cache_directory != _DEFAULT_CACHE_DIR
        rebase: dict[str, tuple[Path, Path, Path]] = {
            # field, default base, default value
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
            "fastembed_model_cache_directory",
        ):
            field_value = getattr(self, field_name)
            if isinstance(field_value, Path):
                field_value.mkdir(parents=True, exist_ok=True)

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

"""Runtime configuration for the arguments dataset pipeline."""

from dataclasses import dataclass
from pathlib import Path

from src.configuration import config as global_config
from src.utils.errors import ConfigurationError


@dataclass
class PipelineConfig:
    """Runtime settings for a pipeline run (CLI overrides global config)."""

    # --- Directories ---
    sources_dir: Path = global_config.dataset_sources_directory
    records_dir: Path = global_config.dataset_records_directory

    # --- Models ---
    backend: str = global_config.llm_backend
    extractor_model: str | None = global_config.extractor_model
    judge_model: str | None = global_config.judge_model

    # --- Pipeline settings ---
    chunk_size: int = global_config.pipeline_chunk_size
    chunk_overlap: int = global_config.pipeline_chunk_overlap
    top_k_links: int = global_config.pipeline_top_k_links
    overwrite: bool = False
    dry_run: bool = False
    limit: int | None = None

    def validated(self) -> "PipelineConfig":
        """Check required settings and prepare directories."""
        if not self.extractor_model:
            msg = (
                "extractor_model is not set. "
                "Provide it via config.toml or --extractor-model."
            )
            raise ConfigurationError(msg)
        if not self.judge_model:
            msg = "judge_model is not set. Provide it via config.toml or --judge-model."
            raise ConfigurationError(msg)
        if self.chunk_overlap >= self.chunk_size:
            msg = "chunk_overlap must be smaller than chunk_size."
            raise ConfigurationError(msg)
        if not self.dry_run:
            self.records_dir.mkdir(parents=True, exist_ok=True)
        return self

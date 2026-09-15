"""Runtime configuration for the arguments dataset pipeline."""

from dataclasses import dataclass
from pathlib import Path

from src.argument_dataset import (
    arg_ds_carriers_native_dir,
    arg_ds_carriers_synthetic_dir,
    arg_ds_compliant_datasets_dir,
    arg_ds_dbs_dir,
    arg_ds_dbs_native_dir,
    arg_ds_gold_dir,
    arg_ds_gold_merged_dir,
)
from src.configuration import config as global_config
from src.utils.errors import ConfigurationError


@dataclass
class PipelineConfig:
    """Runtime settings for a pipeline run (CLI overrides global config)."""

    # --- Directories (legacy single-corpus two-phase pipeline) ---
    sources_dir: Path = arg_ds_carriers_native_dir
    records_dir: Path = arg_ds_gold_dir
    database_dir: Path = arg_ds_gold_dir

    # --- Golden dataset tree ---
    raw_arguments_dir: Path = arg_ds_compliant_datasets_dir
    gold_arguments_dir: Path = arg_ds_gold_dir
    native_sources_dir: Path = arg_ds_carriers_native_dir
    synthetic_sources_dir: Path = arg_ds_carriers_synthetic_dir

    # --- ArgumentDatabases extracted from carriers (ground truth) ---
    extracted_dbs_dir: Path = arg_ds_dbs_dir
    extracted_native_db_dir: Path = arg_ds_dbs_native_dir

    # --- Models ---
    backend: str = global_config.llm_backend
    extractor_model: str | None = global_config.extractor_model
    judge_model: str | None = global_config.judge_model

    # --- Thresholds ---
    premise_contradiction_threshold: float = (
        global_config.premise_contradiction_threshold
    )
    premise_support_threshold: float = global_config.premise_support_threshold
    premise_synthesis_retries: int = global_config.premise_synthesis_retries
    force_premises: bool = False

    # --- Pipeline settings ---
    chunk_size: int = global_config.pipeline_chunk_size
    chunk_overlap: int = global_config.pipeline_chunk_overlap
    top_k_links: int = global_config.pipeline_top_k_links
    overwrite: bool = False
    dry_run: bool = False
    limit: int | None = None

    def validated(self) -> "PipelineConfig":
        """Check required settings."""
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
        return self

    def gold_source_dir(self, source_name: str) -> Path:
        """Return the gold ArgumentDatabase directory of one raw source."""
        return self.gold_arguments_dir / source_name

    def merged_gold_dir(self) -> Path:
        """Return the directory of the merged gold ArgumentDatabase."""
        return self.gold_arguments_dir / arg_ds_gold_merged_dir.name

    def synthetic_batch_dir(self, batch_id: str) -> Path:
        """Return the directory holding one synthetic carrier document batch."""
        return self.synthetic_sources_dir / batch_id

    def extracted_synthetic_db_dir(self, batch_id: str) -> Path:
        """Return where the DB extracted from one synthetic batch is stored."""
        return self.extracted_dbs_dir / f"synthetic_{batch_id}"

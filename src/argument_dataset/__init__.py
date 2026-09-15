"""
Directory tree for the contr-argument golden dataset.

Single source of truth for every path the module needs.
"""

from functools import lru_cache
from pathlib import Path

_DEFAULT_ARG_DATASET_ROOT = Path("./data/argument-dataset")

# --- Root ---
arg_ds_root: Path = _DEFAULT_ARG_DATASET_ROOT

# --- ArgumentDatabase objects extracted FROM carriers (ground truth) ---
arg_ds_dbs_dir: Path = arg_ds_root / "arg_db_objects"
arg_ds_dbs_native_dir: Path = arg_ds_dbs_dir / "native"

# --- Per-source dataframes + gold ArgumentDatabase objects ---
arg_ds_args_dir: Path = arg_ds_root / "arguments"
arg_ds_compliant_datasets_dir: Path = arg_ds_args_dir / "compliant_datasets"
arg_ds_gold_dir: Path = arg_ds_args_dir / "gold"
arg_ds_gold_merged_dir: Path = arg_ds_gold_dir / "merged"

# --- Full carrier documents (real harvested text + generated articles) ---
arg_ds_carriers_dir: Path = arg_ds_root / "argument_carriers"
arg_ds_carriers_imported_dir: Path = arg_ds_carriers_dir / "imported"
arg_ds_carriers_native_dir: Path = arg_ds_carriers_dir / "native"
arg_ds_carriers_synthetic_dir: Path = arg_ds_carriers_dir / "synthetic"


# Ensure full tree exists
@lru_cache(maxsize=1)
def ensure_tree() -> None:
    """Create the dataset directory tree once; later calls are no-ops."""
    for _path in (
        arg_ds_root,
        arg_ds_dbs_dir,
        arg_ds_dbs_native_dir,
        arg_ds_args_dir,
        arg_ds_compliant_datasets_dir,
        arg_ds_gold_dir,
        arg_ds_gold_merged_dir,
        arg_ds_carriers_dir,
        arg_ds_carriers_imported_dir,
        arg_ds_carriers_native_dir,
        arg_ds_carriers_synthetic_dir,
    ):
        _path.mkdir(parents=True, exist_ok=True)


ensure_tree()

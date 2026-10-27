"""
Assemble all source parquets into the single golden arguments pool.

The pool is one table containing argument rows of every origin, raw and
canonical alike, joined by the shared canonical schema. It is the gold
truth the benchmark is evaluated against.
"""

from pathlib import Path

import pandas as pd
from loguru import logger

from src.paths.router import get_data_path


def assemble() -> Path:
    """Merge all source parquets into one pool, validating ids and provenance."""
    datasets_dir = get_data_path(
        subpath="argument-dataset/arguments/datasets", is_file=False
    )
    if not datasets_dir.exists():
        raise FileNotFoundError(
            f"No datasets directory at {datasets_dir}. Run `make fetch-datasets` first."
        )

    parquet_paths = sorted(datasets_dir.glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(
            f"No source parquets in {datasets_dir}. Run `make fetch-datasets` first."
        )

    pool = pd.concat(
        [pd.read_parquet(path) for path in parquet_paths],
        ignore_index=True,
    )

    duplicated = pool["id"][pool["id"].duplicated()].tolist()
    if duplicated:
        raise ValueError(f"Duplicate argument ids in the pool: {sorted(duplicated)}")

    mismatched = pool["id"][
        ~pool.apply(
            lambda row: str(row["id"]).startswith(f"{row['provenance']}:"),
            axis=1,
        )
    ].tolist()
    if mismatched:
        raise ValueError(
            f"Ids not prefixed with their provenance: {sorted(mismatched)}"
        )

    pool_path = get_data_path(subpath="arguments.parquet", is_file=True)
    pool.to_parquet(pool_path, index=False)

    for provenance, count in pool["provenance"].value_counts().items():
        logger.info(f"[{provenance}] {count} rows")
    relations_total = int(pool["relations"].dropna().map(len).sum())
    logger.info(f"Pool: {len(pool)} rows, {relations_total} relations -> {pool_path}")
    return pool_path


def main() -> None:
    """Assemble the golden arguments pool from all fetched sources."""
    assemble()


if __name__ == "__main__":
    main()

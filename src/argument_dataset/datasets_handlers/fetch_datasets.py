"""
Fetch each source, parse it with its adapter, and save the schema-compliant
result to `argument-dataset/arguments/datasets`.

Run with:
    `uv run python -m src.argument_dataset.datasets_handlers.fetch_datasets`
"""

import argparse
from pathlib import Path
from typing import cast

import pandas as pd
from datasets import DatasetDict, load_dataset
from loguru import logger

from src.argument_dataset import arg_ds_compliant_datasets_dir
from src.argument_dataset.datasets_handlers.adapters import HITZ_COUNTER_ARGUMENT
from src.argument_dataset.datasets_handlers.raw_schema import normalize, to_dataframe


def save_canonical_dataset(source_name: str, frame: pd.DataFrame) -> Path:
    """Parse a source's own-shape dataframe and save the canonical result."""
    rows = normalize(source_name, frame)
    path: Path = arg_ds_compliant_datasets_dir / f"{source_name}.parquet"
    to_dataframe(rows).to_parquet(path, index=False)

    logger.info(
        f"[{source_name[:len(source_name)%50]}]\nWrote {len(rows)} "
        f"schema-compliant rows to {path}"
    )
    return path


def fetch_hitz_counter_argument() -> Path:
    """HITZ Dataset (https://huggingface.co/datasets/HiTZ/counter-argument)"""
    dataset: DatasetDict = load_dataset("HiTZ/counter-argument")
    dfs: list[pd.DataFrame] = [
        cast("pd.DataFrame", split.to_pandas()) for split in dataset.values()
    ]
    frame: pd.DataFrame = pd.concat(
        dfs,
        ignore_index=True
    )
    return save_canonical_dataset(HITZ_COUNTER_ARGUMENT, frame)



FETCHERS = {
    HITZ_COUNTER_ARGUMENT: fetch_hitz_counter_argument
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch raw argument datasets.")
    parser.add_argument(
        "--source",
        choices=sorted(FETCHERS),
        action="append",
        default=None,
        help="Source to fetch (repeatable). Defaults to every known source.",
    )
    args = parser.parse_args()

    for source_name in args.source or sorted(FETCHERS):
        FETCHERS[source_name]()


if __name__ == "__main__":
    main()

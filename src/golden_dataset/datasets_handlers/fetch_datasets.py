"""Fetch sources, parse with adapters, save schema-compliant dfs."""

import argparse
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pandas as pd
from datasets import DatasetDict, load_dataset
from loguru import logger

from src.golden_dataset.datasets_handlers.adapter_registry import registry
from src.golden_dataset.datasets_handlers.adapters import DataSource
from src.golden_dataset.schema_models import SchemaArgument
from src.paths.router import get_data_path


def fetch_hitz_counter_argument() -> Path:
    """
    HITZ Dataset: https://huggingface.co/datasets/HiTZ/counter-argument.

    Repeated runs reuse the HF Hub cache; only the adapter pass reruns.
    """
    dataset: DatasetDict = load_dataset("HiTZ/counter-argument")
    dfs: list[pd.DataFrame] = [
        cast("pd.DataFrame", split.to_pandas()) for split in dataset.values()
    ]
    frame: pd.DataFrame = pd.concat(dfs, ignore_index=True)
    return _save_dataset(DataSource.HITZ_COUNTER_ARGUMENT, frame)


def fetch_igor_gld_dataset() -> Path:
    """
    Copy Igor's hand-annotated corpus into the data root, then normalize it.

    The in-repo JSON is the source of record and is never modified; the data
    dir holds the derived copy that all further steps operate on.
    """
    repo_root = Path(__file__).resolve().parents[3]
    source_json = repo_root / "golden_dataset" / "argument-annotations.json"
    copied_json = get_data_path(
        subpath="argument-dataset/sources/igor/argument-annotations.json",
        is_file=True,
    )
    copied_json.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_json, copied_json)

    annotations = json.loads(copied_json.read_text(encoding="utf-8"))

    # Dedup relation pairs (the committed file contains one duplicate).
    unique_pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for relation in annotations["relations"]:
        pair = (str(relation["a"]), str(relation["b"]))
        if pair not in seen:
            seen.add(pair)
            unique_pairs.append(pair)

    counters_by_id: dict[str, list[str]] = {}
    for a, b in unique_pairs:
        counters_by_id.setdefault(a, []).append(b)
        counters_by_id.setdefault(b, []).append(a)

    arguments = annotations["arguments"]
    frame = pd.DataFrame(arguments)
    frame["counter_pairs"] = [
        counters_by_id.get(str(argument_id), []) for argument_id in frame["id"]
    ]
    return _save_dataset(DataSource.IGOR_GLD_DATASET, frame)


def _canonical_dataset_path(source_name: str) -> Path:
    """Path of the canonical parquet for one source."""
    return get_data_path(
        subpath=f"argument-dataset/arguments/datasets/{source_name}.parquet",
        is_file=True,
    )


def _save_dataset(source_name: str, dataframe: pd.DataFrame) -> Path:
    """
    Parse a source's own-shape dataframe and save the argument rows.

    The rows are raw or canonical, depending on the source's adapter.
    """
    path = _canonical_dataset_path(source_name)

    rows = registry.normalize(source_name, dataframe)
    registry.to_dataframe(rows).to_parquet(path, index=False)

    types = ",".join(sorted({row.type for row in rows}))
    logger.info(
        f"[{source_name[: len(source_name) % 50]}]\n"
        f"Wrote {len(rows)} argument rows (types: {types}) to {path}"
    )
    return path


def load_canonical_dataset(source_name: str) -> list[SchemaArgument]:
    """Load one source's parquet into SchemaArgument rows (raw or canonical)."""
    path = _canonical_dataset_path(source_name)
    if not path.exists():
        raise FileNotFoundError(
            f"No canonical dataset for '{source_name}' at {path}. "
            f"Fetch it first: `make fetch-datasets`."
        )
    frame = pd.read_parquet(path)
    return registry.read_canonical(frame)


_fetchers: dict[str, Callable[[], Path]] = {
    DataSource.HITZ_COUNTER_ARGUMENT: fetch_hitz_counter_argument,
    DataSource.IGOR_GLD_DATASET: fetch_igor_gld_dataset,
}


def main() -> None:
    """Fetch every selected source and save its canonical parquet."""
    parser = argparse.ArgumentParser(description="Fetch raw argument datasets.")
    parser.add_argument(
        "--source",
        choices=sorted(_fetchers),
        action="append",
        default=None,
        help="Source to fetch (repeatable). Defaults to every known source.",
    )
    args = parser.parse_args()

    for source_name in args.source or sorted(_fetchers):
        _fetchers[source_name]()


if __name__ == "__main__":
    main()

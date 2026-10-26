"""Module with a Golden Dataset loader."""

# raw samples
# actual matches

import json
from pathlib import Path

from pydantic import BaseModel

from src.data_models.data_models import Argument


class GoldenDataset(BaseModel):
    """A loaded instance of a golde dataset."""

    raw_texts: dict[Path, str]
    arguments: list[Argument]
    paired_arguments: list[tuple[Argument, Argument]]


class GoldenDatasetLoader:
    """Loader for the Golden Dataset."""

    def __init__(self, dataset_path: Path = Path("./golden_dataset")) -> None:
        """
        Ensure the root directory of a dataset exists.

        Args:
            dataset_path (Path): Path to the root of the Golden Dartaset.
                Defaults to Path("./golden_dataset").

        Raises:
            FileNotFoundError: Raised if the provided path to the root of the dataset
                does not exist.
            NotADirectoryError: Raised if the provided path to the root of the dataset
                is not a directory.
        """
        if not dataset_path.exists():
            raise FileNotFoundError(f"There is no such directory: {dataset_path}")
        if not dataset_path.is_dir():
            raise NotADirectoryError("Provided path leads to a non-directory.")
        self.path = dataset_path.expanduser().resolve()

    def load(self) -> GoldenDataset:
        """
        Load a golden dataset from disk.

        Returns:
            GoldenDataset: An initialised instance of a golden dataset.
        """
        annotations_file = self.path / "argument-annotations.json"
        with annotations_file.open("rt") as file:
            data = json.load(file)

        return GoldenDataset(
            raw_texts=self._extract_raw_samples(data=data),
            arguments=self._extract_arguments(data),
            paired_arguments=self._extract_expected_pairs(data),
        )

    def _extract_arguments(self, data: dict) -> list[Argument]:
        arguments: list[Argument] = []

        for item in data["arguments"]:
            claim = item["claim"]["text"]
            arguments.extend(
                Argument(evidence=evidence["text"], claim=claim)
                for evidence in item["evidence"]
            )

        return arguments

    def _extract_expected_pairs(self, data: dict) -> list[tuple[Argument, Argument]]:
        arguments_by_id: dict[int, list[Argument]] = {}

        for item in data["arguments"]:
            claim = item["claim"]["text"]
            arguments_by_id[item["id"]] = [
                Argument(evidence=evidence["text"], claim=claim)
                for evidence in item["evidence"]
            ]

        pairs: list[tuple[Argument, Argument]] = []

        for relation in data["relations"]:
            pairs.extend(
                (a, b)
                for a in arguments_by_id[relation["a"]]
                for b in arguments_by_id[relation["b"]]
            )

        return pairs

    def _extract_raw_samples(self, data: dict) -> dict[Path, str]:
        return {
            self.path / "unannotated_examples" / f"{document['title']}.txt": document[
                "text"
            ]
            for document in data["documents"]
        }

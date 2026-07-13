"""Module with tests of an argument detection dataset class."""

import tempfile
from pathlib import Path

import pytest

# `config` is imported from `argument_detection_dataset` to be monkeypatched.
from src.argument_detection.argument_detection_dataset import (
    ArgumentDetectionDataset,
    config,
)
from src.data_models.data_models import SubsetName


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        (
            "This sentence has too many spaces       .",
            "This sentence has too many spaces .",
        ),
        ("\nAll newlines should be removed.\n\n", "All newlines should be removed."),
        ("No change here!", "No change here!"),
    ],
)
def test_cleaning_sentence(raw: str, clean: str) -> None:
    """Test if sentences are cleaned as expected."""
    assert ArgumentDetectionDataset._clean_sentence(sentence=raw) == clean


@pytest.mark.asyncio
async def test_preparing_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test if preparing a dataset in a location succeeds."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_dir = Path(tmp_dir)

        # Monkeypatch to a temporary directory.
        monkeypatch.setattr(
            config,
            "data_directory",
            temp_dir,
        )

        dataset = ArgumentDetectionDataset()
        await dataset.prepare()

        columns = ["sentence", "is_argument"]
        for subset in SubsetName:
            split_df = dataset.get_split(split=subset)
            assert split_df.shape[1] == len(columns)
            for column in columns:
                assert column in split_df, f"Missing `{column}` in the {subset} split."
                missing_values = split_df[column].isna().sum()
                assert missing_values == 0, (
                    f"Column `{column}` in the {subset} split has {missing_values} "
                    "missing values. Expected no missing values."
                )

            assert split_df.shape[1] > 0

"""Module with tests of an argument detection dataset class."""

import tempfile
from pathlib import Path

import pytest

import src.argument_detection.argument_detection_dataset
from src.argument_detection.argument_detection_dataset import ArgumentDetectionDataset


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
            src.argument_detection.argument_detection_dataset.config,
            "data_directory",
            temp_dir,
        )

        dataset = ArgumentDetectionDataset()
        dataset_path = await dataset.prepare()

        assert dataset_path
        assert dataset_path.exists()

"""Unit tests for the argument framing module."""

import pytest

from src.argument_framing.argument_framing_dataset import ArgumentFramingDataset
from src.argument_framing.xgboost_framer import XGBoostFrameClassifier
from src.utils.errors import ModelNotTrainedError


def test_argument_framing_dataset_initialisation() -> None:
    """Test standard initialisation and default paths."""
    dataset = ArgumentFramingDataset()
    assert dataset.processed_folder_name == "argument_framing"


def test_argument_framing_dataset_prepare_error() -> None:
    """Test if methods guard against unprepared dataset correctly."""
    dataset = ArgumentFramingDataset()
    with pytest.raises(
        ValueError,
        match=r"You have to prepare a dataset before getting one of its splits\.",
    ):
        _ = dataset.get_split("train")


@pytest.mark.asyncio
async def test_xgboost_framer_initialization() -> None:
    """Test if XGBoost framer sets up defaults correctly."""
    mock_dataset = ArgumentFramingDataset()
    framer = XGBoostFrameClassifier(dataset=mock_dataset)

    assert framer.PATH_TO_MODEL.parent.exists()
    assert framer.PATH_TO_VECTORIZER.parent.exists()

    with pytest.raises(ModelNotTrainedError):
        await framer.perform_tuning()

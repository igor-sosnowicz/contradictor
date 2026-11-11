"""Unit tests for the argument framing module."""

import math

import pytest

from src.argument_framing.argument_framer import ArgumentFramer
from src.argument_framing.argument_framing_dataset import ArgumentFramingDataset
from src.argument_framing.xgboost_framer import XGBoostFrameClassifier
from src.data_models.data_models import (
    LABEL_TO_FRAME,
    Argument,
    ArgumentFramerImplementation,
    FramedArgument,
    SubsetName,
)
from src.pipeline.factories.argument_framer_factory import build_argument_framer
from src.utils.errors import ModelNotTrainedError, NotPreparedError


def get_argument_framers() -> list[ArgumentFramer]:
    """
    Provide instances of all argument framers, unprepared and in the PoC mode.

    Returns:
        list[ArgumentFramer]: List of all available argument framer instances,
            Each of them is initialised but unprepared and in the PoC mode.
    """
    return [
        build_argument_framer(implementation, proof_of_concept_mode=True)
        for implementation in ArgumentFramerImplementation
    ]


def test_argument_framing_dataset_initialisation() -> None:
    """Test standard initialisation and default paths."""
    dataset = ArgumentFramingDataset()
    assert dataset.processed_folder_name == "argument_framing"


def test_argument_framing_dataset_prepare_error() -> None:
    """Test if methods guard against unprepared dataset correctly."""
    dataset = ArgumentFramingDataset()
    with pytest.raises(
        NotPreparedError,
        match=r"You have to prepare a dataset before getting one of its splits\.",
    ):
        _ = dataset.get_split(SubsetName.TRAINING)


async def test_xgboost_framer_initialization() -> None:
    """Test if XGBoost framer sets up defaults correctly."""
    mock_dataset = ArgumentFramingDataset()
    framer = XGBoostFrameClassifier(dataset=mock_dataset, proof_of_concept_mode=True)

    assert framer.PATH_TO_MODEL.parent.exists()
    assert framer.PATH_TO_VECTORIZER.parent.exists()

    # Ensure no cached vectorizer exists so perform_tuning() raises immediately.
    framer._vectorizer = None

    with pytest.raises(ModelNotTrainedError):
        await framer.perform_tuning()


@pytest.mark.parametrize("framer", get_argument_framers())
async def test_framing_logic(framer: ArgumentFramer) -> None:
    """Test if an argument framer detects interpretative frame."""
    await framer.prepare()
    arguments = [
        Argument(
            claim="NLP is such a cool field.",
            evidence=(
                "51% of NLP scientists are very happy according to their self-reports."
            ),
        ),
        Argument(
            claim="Ergonomic keyboard improve wrist-associated conditions.",
            evidence="My cats says so.",
        ),
        Argument(
            claim="Rainbows are optical illusions.",
            evidence="Spectrometry proves this using chromatic graphs.",
        ),
    ]
    framed_arguments = framer.frame(arguments=arguments)

    assert len(framed_arguments) == len(arguments)
    for framed_argument, raw_argument in zip(framed_arguments, arguments, strict=True):
        assert isinstance(framed_argument, FramedArgument)

        assert framed_argument.claim == raw_argument.claim, (
            "A claim did not remain unchanged."
        )
        assert framed_argument.evidence == raw_argument.evidence, (
            "A piece of evidence did not remain unchanged."
        )

        assert math.isclose(
            sum(framed_argument.frame_probabilities.values()), 1.0, abs_tol=1e-7
        ), "Frame probabilities do NOT sum up to approximately 1.0."

        assert len(framed_argument.frame_probabilities) == len(LABEL_TO_FRAME)

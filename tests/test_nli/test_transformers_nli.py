"""Module with test for transformer-based NLI."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import huggingface_hub.constants as hf_hub_constants
import pytest

from src.data_models.data_models import NLIPrediction
from src.nli.transformers_nli import TransformersNLI


@pytest.fixture
def transformer_nli(monkeypatch: pytest.MonkeyPatch) -> Generator[TransformersNLI]:
    """Fixture providing a fresh transformer-based implementation of NLI."""
    with (
        tempfile.TemporaryDirectory() as temporary_directory,
    ):
        # Replace a cache directory to enforce the model download.
        monkeypatch.setattr(
            hf_hub_constants,
            "HF_HUB_CACHE",
            Path(temporary_directory) / "hub",
        )
        with TransformersNLI() as nli:
            yield nli


@pytest.mark.parametrize(
    ("batch_size", "should_raise"),
    [
        (-1000, True),
        (-1, True),
        (0, True),
        (1, False),
        (100, False),
    ],
)
def test_batch_size_parameter_validation(
    batch_size: int, *, should_raise: bool
) -> None:
    """Test validation of a batch size parameter."""
    if should_raise:
        with pytest.raises(ValueError, match="Batch size"):
            TransformersNLI(batch_size=batch_size)
    else:
        TransformersNLI(batch_size=batch_size)


@pytest.mark.parametrize(
    ("max_workers", "should_raise"),
    [
        (-3, True),
        (-1, True),
        (0, True),
        (1, False),
        (60, False),
    ],
)
def test_max_workers_parameter_validation(
    max_workers: int, *, should_raise: bool
) -> None:
    """Test validation of a batch size parameter."""
    if should_raise:
        with pytest.raises(ValueError, match="max_workers"):
            TransformersNLI(max_workers=max_workers)
    else:
        TransformersNLI(max_workers=max_workers)


@pytest.mark.asyncio
async def test_nli_inference_returns_valid_result(
    transformer_nli: TransformersNLI,
) -> None:
    """Test if NLI model gets auto-downloaded and can infer."""
    assert isinstance(transformer_nli, TransformersNLI)
    output = await transformer_nli(
        reference_texts=["Hello, world!"], other_texts=["Welcome, my dear world."]
    )
    assert output
    assert isinstance(output, list)
    assert len(output) == 1
    assert isinstance(output[0][0], NLIPrediction)  # A result is an NLI prediction.


@pytest.mark.asyncio
async def test_output_shape(transformer_nli: TransformersNLI) -> None:
    """Test if shape of the output of the model is valid."""
    reference_texts = ["Contradictor here", "It works.", "It's the best."]
    other_texts = ["It's the worst.", "Linux is the best kernel."]

    output = await transformer_nli(reference_texts, other_texts)
    expected_reference_texts = len(reference_texts)
    expected_other_texts = len(other_texts)

    assert len(output) == expected_reference_texts
    for i in range(expected_reference_texts):
        assert len(output[i]) == expected_other_texts

    # Flatten.
    flat_output = [item for row in output for item in row]

    # Verify total number of predictions.
    assert len(flat_output) == expected_reference_texts * expected_other_texts

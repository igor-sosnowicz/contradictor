"""Tests for evidence extractor."""

import asyncio

import numpy as np

from src.argument_detection.xgboost_evidence_extractor import (
    XGBoostEvidenceExtractor,
)


class FakeDataset:
    """Fake dataset."""

    async def prepare(self) -> None:
        """Prepare fake dataset."""


class FakeModel:
    """Fake classifier with controlled predictions."""

    def __init__(
        self,
        probabilities: list[float],
    ) -> None:
        """
        Initialize fake classifier.

        Args:
            probabilities:
                Probability of the evidence class for each input row.
        """
        self._probabilities = probabilities
        self.predict_proba_calls = 0
        self.last_input: np.ndarray | None = None

    def predict_proba(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Return controlled probabilities.

        Args:
            x:
                Input feature matrix.

        Returns:
            Probability predictions for each input row.
        """
        self.predict_proba_calls += 1
        self.last_input = x

        if len(x) != len(self._probabilities):
            raise AssertionError(
                "Unexpected number of model inputs: "
                f"expected {len(self._probabilities)}, "
                f"got {len(x)}",
            )

        return np.array(
            [[1.0 - probability, probability] for probability in self._probabilities],
        )


def create_extractor(
    model: FakeModel,
) -> XGBoostEvidenceExtractor:
    """Create an evidence extractor with a fake model."""
    extractor = XGBoostEvidenceExtractor(
        FakeDataset(),
        proof_of_concept_mode=True,
    )
    extractor._model = model

    return extractor


def test_evidence_extractor_returns_evidence() -> None:
    """Verify evidence sentences are returned."""
    model = FakeModel([0.9])
    extractor = create_extractor(model)

    result = asyncio.run(
        extractor.extract_evidence(
            "Cats are smart.",
            "Cats learn quickly.",
        ),
    )

    assert result == ["Cats learn quickly."]
    assert model.predict_proba_calls == 1
    assert model.last_input is not None
    assert model.last_input.shape[0] == 1


def test_evidence_extractor_filters_non_evidence_candidates() -> None:
    """Verify only candidates classified as evidence are returned."""
    expected_model_calls = 1

    model = FakeModel([0.9, 0.1])
    extractor = create_extractor(model)

    result = asyncio.run(
        extractor.extract_evidence(
            "Cats are smart.",
            "Cats learn quickly. Dogs are loyal.",
        ),
    )

    assert result == ["Cats learn quickly."]
    assert model.predict_proba_calls == expected_model_calls


def test_evidence_extractor_excludes_claim() -> None:
    """Verify the claim itself is not returned as evidence."""
    model = FakeModel([0.9])
    extractor = create_extractor(model)

    result = asyncio.run(
        extractor.extract_evidence(
            "Cats are smart.",
            "Cats are smart. Cats learn quickly.",
        ),
    )

    assert result == ["Cats learn quickly."]
    assert model.predict_proba_calls == 1
    assert model.last_input is not None
    assert model.last_input.shape[0] == 1


def test_evidence_extractor_returns_empty_list_when_text_contains_only_claim() -> None:
    """Verify no evidence is returned when only the claim is present."""
    model = FakeModel([])
    extractor = create_extractor(model)

    result = asyncio.run(
        extractor.extract_evidence(
            "Cats are smart.",
            "Cats are smart.",
        ),
    )

    assert result == []
    assert model.predict_proba_calls == 0
    assert model.last_input is None


def test_evidence_extractor_filters_low_probability_evidence() -> None:
    """Verify candidates below the classification threshold are excluded."""
    model = FakeModel([0.1])
    extractor = create_extractor(model)

    result = asyncio.run(
        extractor.extract_evidence(
            "Cats are smart.",
            "Cats learn quickly.",
        ),
    )

    assert result == []
    assert model.predict_proba_calls == 1
    assert model.last_input is not None
    assert model.last_input.shape[0] == 1

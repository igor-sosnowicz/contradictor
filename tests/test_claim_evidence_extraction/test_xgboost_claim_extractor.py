"""Tests for claim extractor."""

import asyncio

import numpy as np
import pytest

from src.argument_detection.xgboost_claim_extractor import (
    XGBoostClaimExtractor,
)


class FakeDataset:
    """Fake dataset."""

    async def prepare(self) -> None:
        """Prepare fake dataset."""


class FakeModel:
    """Fake classifier."""

    def __init__(self, claim_probability: float) -> None:
        """
        Initialize fake classifier.

        Args:
            claim_probability:
                Probability assigned to the claim class.
        """
        self.claim_probability = claim_probability
        self.predict_proba_calls = 0
        self.last_input: np.ndarray | None = None

    def predict_proba(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Return fake probabilities.

        Args:
            x:
                Input features.

        Returns:
            Prediction probabilities.
        """
        self.predict_proba_calls += 1
        self.last_input = x

        return np.array(
            [[1.0 - self.claim_probability, self.claim_probability]],
        )


class FakeEmbedder:
    """Fake embedder."""

    def __init__(self) -> None:
        """Initialize fake embedder."""
        self.embed_calls = 0
        self.last_texts: list[str] | None = None

    def embed(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """
        Return fake embeddings.

        Args:
            texts:
                Texts to embed.

        Returns:
            Fake embeddings.
        """
        self.embed_calls += 1
        self.last_texts = texts

        return np.array([[1.0, 2.0, 3.0]])


def create_extractor(
    monkeypatch: pytest.MonkeyPatch,
    model: FakeModel,
    embedder: FakeEmbedder,
) -> XGBoostClaimExtractor:
    """Create extractor with fake dependencies."""
    extractor = XGBoostClaimExtractor(
        FakeDataset(),
        proof_of_concept_mode=True,
    )

    extractor._model = model
    extractor._embedder = embedder

    monkeypatch.setattr(
        extractor,
        "_sentence_splitter",
        lambda text: ["Cats are intelligent."],
    )

    async def fake_initialise_model() -> None:
        """Skip model initialization."""

    monkeypatch.setattr(
        extractor,
        "_initialise_model",
        fake_initialise_model,
    )

    return extractor


def test_claim_extractor_returns_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that a sentence classified as a claim is returned."""
    model = FakeModel(claim_probability=0.8)
    embedder = FakeEmbedder()

    extractor = create_extractor(
        monkeypatch,
        model,
        embedder,
    )

    result = asyncio.run(
        extractor.extract_claims("Cats are intelligent."),
    )

    assert result == ["Cats are intelligent."]

    assert embedder.embed_calls == 1
    assert embedder.last_texts == ["Cats are intelligent."]

    assert model.predict_proba_calls == 1
    assert model.last_input is not None
    np.testing.assert_array_equal(
        model.last_input,
        np.array([[1.0, 2.0, 3.0]]),
    )


def test_claim_extractor_rejects_non_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that a sentence classified as a non-claim is rejected."""
    model = FakeModel(claim_probability=0.2)
    embedder = FakeEmbedder()

    extractor = create_extractor(
        monkeypatch,
        model,
        embedder,
    )

    result = asyncio.run(
        extractor.extract_claims("Cats are intelligent."),
    )

    assert result == []

    assert embedder.embed_calls == 1
    assert embedder.last_texts == ["Cats are intelligent."]

    assert model.predict_proba_calls == 1
    assert model.last_input is not None
    np.testing.assert_array_equal(
        model.last_input,
        np.array([[1.0, 2.0, 3.0]]),
    )

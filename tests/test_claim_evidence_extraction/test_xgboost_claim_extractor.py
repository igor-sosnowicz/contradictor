"""Tests for claim extractor."""

import asyncio

import numpy as np
import pytest

from src.argument_detection.xgboost_claim_extractor import (
    XGBoostClaimExtractor,
)


class FakeDataset:
    """Fake dataset."""

    async def prepare(
        self,
    ) -> None:
        """Prepare fake dataset."""


class FakeModel:
    """Fake classifier."""

    def predict_proba(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Return fake probabilities.

        Args:
            x (np.ndarray):
                Input features.

        Returns:
            np.ndarray:
                Prediction probabilities.
        """
        return np.array([[0.2, 0.8]])


class FakeEmbedder:
    """Fake embedder."""

    def embed(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """
        Return fake embeddings.

        Args:
            texts (list[str]):
                Input texts.

        Returns:
            np.ndarray:
                Fake embeddings.
        """
        return np.array([[1.0, 2.0, 3.0]])


def test_claim_extractor_returns_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify claim extraction."""
    extractor = XGBoostClaimExtractor(
        FakeDataset(),
        proof_of_concept_mode=True,
    )

    extractor._model = FakeModel()
    extractor._embedder = FakeEmbedder()

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

    result = asyncio.run(
        extractor.extract_claims("Cats are intelligent."),
    )

    assert result == ["Cats are intelligent."]

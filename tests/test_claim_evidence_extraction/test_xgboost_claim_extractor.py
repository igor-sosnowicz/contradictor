"""Tests for claim extractor."""

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
        """
        Prepare fake dataset.

        Returns:
            None.
        """


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
        return np.array(
            [
                [
                    0.2,
                    0.8,
                ]
            ]
        )


def test_claim_extractor_returns_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify claim extraction."""
    extractor = XGBoostClaimExtractor(
        FakeDataset(),
        proof_of_concept_mode=True,
    )

    extractor._model = FakeModel()

    monkeypatch.setattr(
        extractor,
        "_embed_texts",
        lambda texts: np.array([[1, 2, 3]]),
    )

    import asyncio

    result = asyncio.run(extractor.extract_claims("Cats are intelligent."))

    assert result == ["Cats are intelligent."]

"""Tests for evidence extractor."""

import numpy as np
import pytest

from src.argument_detection.xgboost_evidence_extractor import (
    XGBoostEvidenceExtractor,
)


class FakeDataset:
    """Fake dataset."""

    async def prepare(
        self,
    ) -> None:
        """
        Prepare fake dataset.

        Used only for extractor tests.
        """


class FakeModel:
    """Fake classifier."""

    def predict_proba(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Return fake prediction probabilities.

        Args:
            x:
                Input feature matrix.

        Returns:
            Probability predictions.
        """
        return np.array(
            [
                [
                    0.1,
                    0.9,
                ]
            ]
        )


def test_evidence_extractor_returns_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify evidence extraction."""
    extractor = XGBoostEvidenceExtractor(
        FakeDataset(),
        proof_of_concept_mode=True,
    )

    extractor._model = FakeModel()

    monkeypatch.setattr(
        extractor,
        "_create_features",
        lambda claims, evidence: np.array([[1, 2]]),
    )

    import asyncio

    result = asyncio.run(
        extractor.extract_evidence(
            "Cats are smart.",
            "Cats learn quickly.",
        )
    )

    assert result == ["Cats learn quickly."]

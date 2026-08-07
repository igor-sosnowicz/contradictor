"""Tests for evidence extractor."""

import asyncio

import numpy as np
import pandas as pd
import pytest

from src.argument_detection.xgboost_evidence_extractor import (
    XGBoostEvidenceExtractor,
)
from src.data_models.data_models import SubsetName


class FakeDataset:
    """Fake dataset."""

    async def prepare(self) -> None:
        """Prepare fake dataset."""

    def get_evidence_split(
        self,
        split: SubsetName,
        max_samples: int | None = None,
    ) -> pd.DataFrame:
        """
        Return a fake evidence extraction dataset split.

        Args:
            split (SubsetName):
                Dataset subset to retrieve.

            max_samples (int | None):
                Optional maximum number of rows returned.

        Returns:
            pd.DataFrame:
                Fake evidence extraction samples.
        """
        return pd.DataFrame(
            {
                "claim": [
                    "Cats are smart.",
                    "Dogs are loyal.",
                    "Birds can fly.",
                    "Fish live underwater.",
                    "Horses are strong.",
                ],
                "evidence": [
                    "Cats learn quickly.",
                    "Dogs help humans.",
                    "Birds use wings.",
                    "Fish breathe through gills.",
                    "Horses have powerful muscles.",
                ],
                "is_evidence": [
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
            },
        )


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
                ],
            ],
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

    async def fake_initialise_model() -> None:
        """Skip model initialization."""

    monkeypatch.setattr(
        extractor,
        "_initialise_model",
        fake_initialise_model,
    )

    monkeypatch.setattr(
        extractor,
        "_create_features",
        lambda claims, evidence: np.array([[1, 2]]),
    )

    result = asyncio.run(
        extractor.extract_evidence(
            "Cats are smart.",
            "Cats learn quickly.",
        )
    )

    assert result == ["Cats learn quickly."]

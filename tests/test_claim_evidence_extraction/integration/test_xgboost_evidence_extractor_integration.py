"""Integration tests for the evidence extractior."""

import pandas as pd
import pytest
import pytest_asyncio
from xgboost import XGBClassifier

from src.argument_detection.argument_detection_dataset import ArgumentDetectionDataset
from src.argument_detection.xgboost_evidence_extractor import XGBoostEvidenceExtractor
from src.data_models.data_models import SubsetName


class MiniIntegrationDataset(ArgumentDetectionDataset):
    """
    Provides a small, deterministic dataset for integration tests.

    The dataset is fully local and does not require access to Kaggle or any
    external network resources.
    """

    def __init__(self) -> None:
        """Initialize the deterministic integration-test dataset."""

    async def prepare(self) -> None:
        """Mark the dataset as prepared."""
        self._is_prepared = True

    def get_evidence_split(
        self,
        split: SubsetName,
        max_samples: int | None = None,
    ) -> pd.DataFrame:
        """Return a deterministic claim-evidence classification dataset."""
        data = pd.DataFrame(
            {
                "claim": [
                    "Cats are smart.",
                    "Cats are smart.",
                    "Dogs are loyal.",
                    "Dogs are loyal.",
                    "Birds can fly.",
                    "Birds can fly.",
                    "Fish live underwater.",
                    "Fish live underwater.",
                    "Trees need water.",
                    "Trees need water.",
                ],
                "evidence": [
                    "Cats learn quickly.",
                    "Cars need fuel.",
                    "Dogs help humans.",
                    "The sky is blue.",
                    "Birds use wings.",
                    "Books contain information.",
                    "Fish breathe through gills.",
                    "Horses are strong.",
                    "Trees absorb water.",
                    "Trees absorb water.",
                ],
                "is_evidence": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            }
        )
        return data.head(max_samples) if max_samples is not None else data


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def evidence_extractor() -> XGBoostEvidenceExtractor:
    """
    Create and train one real evidence extractor for the test module.

    The real embedder and XGBoost classifier are used. The trained model is
    reused by all tests in this module.
    """
    dataset = MiniIntegrationDataset()
    extractor = XGBoostEvidenceExtractor(
        dataset,
        proof_of_concept_mode=True,
    )

    model = await extractor._train_model()

    assert isinstance(model, XGBClassifier)

    extractor._config = extractor._config.model_copy(
        update={
            "threshold": extractor._config.threshold.model_copy(update={"claim": 0.5})
        }
    )

    return extractor


@pytest.fixture
def low_threshold_extractor(
    evidence_extractor: XGBoostEvidenceExtractor,
) -> XGBoostEvidenceExtractor:
    """Configure the extractor with a permissive evidence threshold."""
    evidence_extractor._config = evidence_extractor._config.model_copy(
        update={
            "threshold": evidence_extractor._config.threshold.model_copy(
                update={"claim": 0.5}
            )
        }
    )
    return evidence_extractor


@pytest.mark.asyncio
async def test_evidence_extractor_returns_evidence(
    low_threshold_extractor: XGBoostEvidenceExtractor,
) -> None:
    """Verify that a supporting evidence sentence is returned."""
    result = await low_threshold_extractor.extract_evidence(
        "Cats are smart.",
        "Cats learn quickly.",
    )

    assert result
    assert "Cats learn quickly." in result


@pytest.mark.asyncio
async def test_evidence_extractor_filters_non_evidence_candidates(
    low_threshold_extractor: XGBoostEvidenceExtractor,
) -> None:
    """Verify that non-evidence candidates are excluded from the result."""
    result = await low_threshold_extractor.extract_evidence(
        "Cats are smart.",
        "Cats learn quickly. Cars need fuel.",
    )

    assert "Cats learn quickly." in result
    assert "Cars need fuel." not in result


@pytest.mark.asyncio
async def test_evidence_extractor_excludes_claim(
    low_threshold_extractor: XGBoostEvidenceExtractor,
) -> None:
    """Verify that the claim itself is excluded from the result."""
    result = await low_threshold_extractor.extract_evidence(
        "Cats are smart.",
        "Cats are smart. Cats learn quickly.",
    )

    assert "Cats are smart." not in result
    assert "Cats learn quickly." in result


@pytest.mark.asyncio
async def test_evidence_extractor_returns_empty_list_when_text_contains_only_claim(
    low_threshold_extractor: XGBoostEvidenceExtractor,
) -> None:
    """Verify that no evidence is returned when the text contains only claim."""
    result = await low_threshold_extractor.extract_evidence(
        "Cats are smart.",
        "Cats are smart.",
    )

    assert result == []


@pytest.mark.asyncio
async def test_evidence_extractor_respects_classification_threshold(
    evidence_extractor: XGBoostEvidenceExtractor,
) -> None:
    """Verify that candidates below the threshold are excluded."""
    evidence_extractor._config = evidence_extractor._config.model_copy(
        update={
            "threshold": evidence_extractor._config.threshold.model_copy(
                update={"claim": 0.99}
            )
        }
    )

    result = await evidence_extractor.extract_evidence(
        "Cats are smart.",
        "Cats learn quickly.",
    )

    assert result == []

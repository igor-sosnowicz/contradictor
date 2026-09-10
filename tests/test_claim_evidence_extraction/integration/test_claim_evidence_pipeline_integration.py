"""Integration tests for the claim-evidence extraction pipeline."""

import pandas as pd
import pytest
import pytest_asyncio
from xgboost import XGBClassifier

from src.argument_detection.argument_detection_dataset import (
    ArgumentDetectionDataset,
)
from src.argument_detection.claim_evidence_extractor import (
    ClaimEvidenceExtractor,
)
from src.argument_detection.xgboost_claim_extractor import (
    XGBoostClaimExtractor,
)
from src.argument_detection.xgboost_evidence_extractor import (
    XGBoostEvidenceExtractor,
)
from src.data_models.data_models import ClaimEvidencePair, SubsetName


class MiniPipelineDataset(ArgumentDetectionDataset):
    """Small deterministic dataset for pipeline integration tests."""

    def __init__(self) -> None:
        """Initialize the deterministic test dataset."""

    async def prepare(self) -> None:
        """Mark dataset as prepared."""
        self._is_prepared = True

    def get_claim_split(
        self,
        split: SubsetName,
        max_samples: int | None = None,
    ) -> pd.DataFrame:
        """Return a deterministic claim classification dataset."""
        data = pd.DataFrame(
            {
                "sentence": [
                    "Cats are intelligent animals.",
                    "The sky is blue.",
                    "Dogs are loyal animals.",
                    "Water freezes at zero degrees.",
                    "Cats learn quickly.",
                    "Books contain information.",
                    "Birds can fly.",
                    "Cars need fuel.",
                    "Trees need water.",
                    "The sun provides light.",
                ],
                "is_claim": [
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                ],
            },
        )
        return data.head(max_samples) if max_samples is not None else data

    def get_evidence_split(
        self,
        split: SubsetName,
        max_samples: int | None = None,
    ) -> pd.DataFrame:
        """Return a deterministic claim-evidence classification dataset."""
        data = pd.DataFrame(
            {
                "claim": [
                    "Cats are intelligent animals.",
                    "Cats are intelligent animals.",
                    "Dogs are loyal animals.",
                    "Dogs are loyal animals.",
                    "Cats learn quickly.",
                    "Cats learn quickly.",
                    "Birds can fly.",
                    "Birds can fly.",
                    "Trees need water.",
                    "Trees need water.",
                ],
                "evidence": [
                    "Cats learn quickly.",
                    "Cars need fuel.",
                    "Dogs help humans.",
                    "The sky is blue.",
                    "Cats are intelligent animals.",
                    "Horses are strong.",
                    "Birds use wings.",
                    "Books contain information.",
                    "Trees absorb water.",
                    "Cars need fuel.",
                ],
                "is_evidence": [
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                ],
            },
        )
        return data.head(max_samples) if max_samples is not None else data


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def pipeline() -> ClaimEvidenceExtractor:
    """
    Create a real, trained claim-evidence extraction pipeline.

    The fixture trains both XGBoost extractors once and reuses them across
    all tests in the module.
    """
    dataset = MiniPipelineDataset()

    claim_extractor = XGBoostClaimExtractor(
        dataset,
        proof_of_concept_mode=True,
    )

    evidence_extractor = XGBoostEvidenceExtractor(
        dataset,
        proof_of_concept_mode=True,
    )

    claim_model = await claim_extractor._train_model()
    evidence_model = await evidence_extractor._train_model()

    assert isinstance(claim_model, XGBClassifier)
    assert isinstance(evidence_model, XGBClassifier)

    claim_extractor._config = claim_extractor._config.model_copy(
        update={
            "threshold": claim_extractor._config.threshold.model_copy(
                update={"claim": 0.5},
            ),
        },
    )

    evidence_extractor._config = evidence_extractor._config.model_copy(
        update={
            "threshold": evidence_extractor._config.threshold.model_copy(
                update={"claim": 0.5},
            ),
        },
    )

    return ClaimEvidenceExtractor(
        claim_extractor=claim_extractor,
        evidence_extractor=evidence_extractor,
    )


@pytest.mark.asyncio
async def test_pipeline_extracts_claim_evidence_pairs(
    pipeline: ClaimEvidenceExtractor,
) -> None:
    """Verify that the real pipeline extracts claim-evidence pairs."""
    result = await pipeline.extract_pairs(
        "Cats are intelligent animals. Cats learn quickly.",
    )

    assert result
    assert all(isinstance(pair, ClaimEvidencePair) for pair in result)

    assert any(
        pair.claim == "Cats are intelligent animals."
        and pair.evidence == "Cats learn quickly."
        for pair in result
    )

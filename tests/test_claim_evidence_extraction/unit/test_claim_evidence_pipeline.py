"""Unit tests for the claim-evidence extraction pipeline."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.argument_detection.claim_evidence_extractor import (
    ClaimEvidenceExtractor,
)
from src.data_models.data_models import Argument


@pytest.fixture
def mock_pipeline() -> ClaimEvidenceExtractor:
    """Create a pipeline with mocked components for unit tests."""
    claim_extractor = MagicMock()
    evidence_extractor = MagicMock()

    claim_extractor.extract_claims_batch = AsyncMock()
    evidence_extractor.extract_evidence_batch = AsyncMock()

    return ClaimEvidenceExtractor(
        claim_extractor=claim_extractor,
        evidence_extractor=evidence_extractor,
    )


@pytest.mark.asyncio
async def test_pipeline_extracts_pairs(
    mock_pipeline: ClaimEvidenceExtractor,
) -> None:
    """Verify that the pipeline orchestrates component steps to extract pairs."""
    mock_pipeline._claim_extractor.extract_claims_batch = AsyncMock(
        return_value=[["Cats are smart."]]
    )
    mock_pipeline._evidence_extractor.extract_evidence_batch = AsyncMock(
        return_value=[["Cats learn quickly."]]
    )

    result_batch = await mock_pipeline._process_batch(["Some full document text"])
    result = result_batch

    assert len(result) == 1
    assert isinstance(result, Argument)
    assert result.claim == "Cats are smart."
    assert result.evidence == "Cats learn quickly."

    mock_pipeline._claim_extractor.extract_claims_batch.assert_called_once_with(
        ["Some full document text"]
    )
    mock_pipeline._evidence_extractor.extract_evidence_batch.assert_called_once_with(
        claims=["Cats are smart."],
        texts=["Some full document text"],
    )


@pytest.mark.asyncio
async def test_pipeline_returns_empty_when_no_claims(
    mock_pipeline: ClaimEvidenceExtractor,
) -> None:
    """Verify that an empty list is returned if no claims are detected."""
    mock_pipeline._claim_extractor.extract_claims_batch = AsyncMock(return_value=[[]])

    result_batch = await mock_pipeline._process_batch(["Text without claims"])
    result = result_batch

    assert result == []
    mock_pipeline._evidence_extractor.extract_evidence_batch.assert_not_called()

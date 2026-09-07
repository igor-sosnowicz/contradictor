"""Tests for claim extractor module."""

import asyncio

from src.argument_detection.claim_evidence_extractor import (
    ClaimEvidenceExtractor,
)
from src.data_models.data_models import ClaimEvidencePair


class FakeClaimExtractor:
    """Fake claim extractor."""

    async def extract_claims(
        self,
        text: str,
    ) -> list[str]:
        """Return a predefined claim."""
        return ["Cats are intelligent animals."]


class FakeEvidenceExtractor:
    """Fake evidence extractor."""

    async def extract_evidence(
        self,
        claim: str,
        text: str,
    ) -> list[str]:
        """Return predefined evidence for a claim."""
        return ["Cats learn quickly."]


def test_pipeline_extracts_pairs() -> None:
    """Verify pipeline creates claim-evidence pairs."""
    pipeline = ClaimEvidenceExtractor(
        claim_extractor=FakeClaimExtractor(),
        evidence_extractor=FakeEvidenceExtractor(),
    )

    result = asyncio.run(
        pipeline.extract_pairs("Cats are intelligent animals."),
    )

    assert len(result) == 1
    assert isinstance(result[0], ClaimEvidencePair)
    assert result[0].claim == "Cats are intelligent animals."
    assert result[0].evidence == "Cats learn quickly."


def test_pipeline_returns_empty_when_no_claims() -> None:
    """Verify empty result when claims are missing."""

    class EmptyClaimExtractor:
        """Fake extractor returning no claims."""

        async def extract_claims(
            self,
            text: str,
        ) -> list[str]:
            """Return an empty list of claims."""
            return []

    pipeline = ClaimEvidenceExtractor(
        claim_extractor=EmptyClaimExtractor(),
        evidence_extractor=FakeEvidenceExtractor(),
    )

    result = asyncio.run(pipeline.extract_pairs("text"))

    assert result == []

"""
Pipeline for extracting claim-evidence pairs from text.

The module combines claim detection and evidence extraction into a single
processing flow:

    text
      -> claim extractor
      -> claims
      -> evidence extractor
      -> claim-evidence pairs
"""

from src.argument_detection.xgboost_claim_extractor import XGBoostClaimExtractor
from src.argument_detection.xgboost_evidence_extractor import XGBoostEvidenceExtractor
from src.data_models.data_models import ClaimEvidencePair


class ClaimEvidenceExtractor:
    """
    Pipeline for extracting claim-evidence pairs from documents.

    The extractor combines two components:
    - ClaimExtractor for identifying claims,
    - EvidenceExtractor for finding supporting evidence.

    The extraction process consists of:
    1. extracting claims from input text,
    2. finding evidence related to each claim,
    3. creating claim-evidence pair objects.
    """

    def __init__(
        self,
        claim_extractor: XGBoostClaimExtractor,
        evidence_extractor: XGBoostEvidenceExtractor,
    ) -> None:
        """
        Initialize the claim-evidence extraction pipeline.

        Args:
            claim_extractor (ClaimExtractor):
                Component responsible for extracting claims from text.

            evidence_extractor (EvidenceExtractor):
                Component responsible for extracting evidence supporting claims.

        Returns:
            None
        """
        self._claim_extractor = claim_extractor
        self._evidence_extractor = evidence_extractor

    async def extract_pairs(
        self,
        text: str,
    ) -> list[ClaimEvidencePair]:
        """
        Extract claim-evidence pairs from input text.

        The method first extracts claims from the document and then searches
        for evidence supporting each extracted claim.

        Args:
            text (str):
                Input essay or document.

        Returns:
            list[ClaimEvidencePair]:
                List of extracted claim-evidence pairs.

        Raises:
            Exception:
                Propagates exceptions raised by claim or evidence extractors.
        """
        claims = await self._claim_extractor.extract_claims(text)
        pairs: list[ClaimEvidencePair] = []
        for claim in claims:
            evidences = await self._evidence_extractor.extract_evidence(
                claim,
                text,
            )
            pairs.extend(
                ClaimEvidencePair(
                    claim=claim,
                    evidence=evidence,
                )
                for evidence in evidences
            )
        return pairs

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

import asyncio

from src.argument_detection.xgboost_claim_extractor import XGBoostClaimExtractor
from src.argument_detection.xgboost_evidence_extractor import (
    XGBoostEvidenceExtractor,
)
from src.data_models.data_models import Argument
from src.pipeline.argument_extractor import ArgumentExtractor


class ClaimEvidenceExtractor(ArgumentExtractor):
    """
    Pipeline for extracting claim-evidence pairs from documents.

    The extractor combines two components:
    - ClaimExtractor for identifying claims,
    - EvidenceExtractor for finding supporting evidence.
    """

    def __init__(
        self,
        claim_extractor: XGBoostClaimExtractor,
        evidence_extractor: XGBoostEvidenceExtractor,
    ) -> None:
        """Initialize the claim-evidence extraction pipeline."""
        self._claim_extractor = claim_extractor
        self._evidence_extractor = evidence_extractor

    def __call__(self, texts: list[str]) -> list[list[Argument]]:
        """
        Extract arguments (claim+evidence) from multiple texts.

        Args:
            texts (list[str]): Texts, from which arguments should be extracted.

        Returns:
            list[list[Argument]]: Lists of extracted arguments.
                One per input document. Order is preserved.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with asyncio.Runner() as runner:
                return runner.run(self._process_batch(texts))
        else:
            return asyncio.run(self._process_batch(texts))

    async def _process_batch(self, texts: list[str]) -> list[list[Argument]]:
        """Process documents in a fully vectorized/batched manner."""
        all_docs_claims: list[
            list[str]
        ] = await self._claim_extractor.extract_claims_batch(texts)

        flat_claims: list[str] = []
        flat_texts: list[str] = []
        doc_indices: list[int] = []

        for doc_idx, claims in enumerate(all_docs_claims):
            for claim in claims:
                flat_claims.append(claim)
                flat_texts.append(texts[doc_idx])
                doc_indices.append(doc_idx)

        if not flat_claims:
            return [[] for _ in texts]

        flat_evidences: list[
            list[str]
        ] = await self._evidence_extractor.extract_evidence_batch(
            claims=flat_claims,
            texts=flat_texts,
        )

        results: list[list[Argument]] = [[] for _ in texts]

        for i, evidences in enumerate(flat_evidences):
            target_doc_idx = doc_indices[i]
            current_claim = flat_claims[i]

            for evidence in evidences:
                results[target_doc_idx].append(
                    Argument(claim=current_claim, evidence=evidence)
                )

        return results

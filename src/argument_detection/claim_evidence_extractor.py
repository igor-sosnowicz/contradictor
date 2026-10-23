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
from typing import override

from src.argument_detection.argument_extractor import ArgumentExtractor
from src.argument_detection.xgboost_claim_extractor import XGBoostClaimExtractor
from src.argument_detection.xgboost_evidence_extractor import (
    XGBoostEvidenceExtractor,
)
from src.data_models.data_models import Argument


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
        """
        Initialise the claim-evidence extraction pipeline.

        Args:
            claim_extractor (XGBoostClaimExtractor): Component responsible for
                extracting claims from text.
            evidence_extractor (XGBoostEvidenceExtractor): Component
                responsible for extracting evidence supporting claims.
        """
        self._claim_extractor = claim_extractor
        self._evidence_extractor = evidence_extractor

    @override
    def __call__(self, texts: list[str]) -> list[list[Argument]]:
        """
        Provide a synchronous entry point for ordinary scripts.

        Args:
            texts (list[str]): Texts, from which arguments should be extracted.

        Returns:
            list[list[Argument]]: Lists of extracted arguments.
                One per input document. Order is preserved.

        Raises:
            RuntimeError: If called from inside a running event loop.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._process_batch(texts))

        raise RuntimeError(
            "__call_sync__() cannot be called from a running event loop; "
            "use 'await extractor(...)' instead"
        )

    async def extract_batch(self, texts: list[str]) -> list[list[Argument]]:
        """
        Extract arguments from multiple texts asynchronously.

        Args:
            texts (list[str]): Texts, from which arguments should be extracted.

        Returns:
            list[list[Argument]]: Lists of extracted arguments.
                One per input document. Order is preserved.
        """
        return await self._process_batch(texts)

    # pylint: disable=too-many-locals
    # Reason: Managing batch document indexes, flat claims, flat texts,
    # and mapped results simultaneously in a vectorized
    # pipeline naturally requires a higher local variable count.
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

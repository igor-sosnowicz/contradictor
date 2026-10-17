"""Module with an argument extractor factory function."""

from src.argument_detection.argument_detection_dataset import ArgumentDetectionDataset
from src.argument_detection.claim_evidence_extractor import ClaimEvidenceExtractor
from src.argument_detection.xgboost_claim_extractor import XGBoostClaimExtractor
from src.argument_detection.xgboost_evidence_extractor import XGBoostEvidenceExtractor
from src.configuration import config
from src.data_models.data_models import ArgumentExtractorImplementation
from src.pipeline.argument_extractor import ArgumentExtractor


def build_argument_extractor(
    extractor_implementation: ArgumentExtractorImplementation | None = None,
) -> ArgumentExtractor:
    """
    Construct a ready-made argument extractor from pre-configured implementations.

    Args:
        extractor_implementation (ArgumentExtractorImplementation): Choice of a
            pre-configured implementation.

    Returns:
        ArgumentExtractor: An initialised instance of the argument extractor.

    Raises:
        NotImplementedError: Raised if not implemented variant was requested.
    """
    extractor_implementation = extractor_implementation or config.argument_extractor

    match extractor_implementation:
        case ArgumentExtractorImplementation.XGBOOST:
            dataset = ArgumentDetectionDataset()

            claim_extractor = XGBoostClaimExtractor(dataset=dataset)
            evidence_extractor = XGBoostEvidenceExtractor(dataset=dataset)

            return ClaimEvidenceExtractor(
                claim_extractor=claim_extractor,
                evidence_extractor=evidence_extractor,
            )

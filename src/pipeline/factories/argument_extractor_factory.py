"""Module with an argument extractor factory function."""

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
        case ArgumentExtractorImplementation.NOT_IMPLEMENTED:
            raise NotImplementedError("No argument extractor is implemented yet.")

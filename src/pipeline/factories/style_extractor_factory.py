"""Module with a style extractor factory function."""

from src.configuration import config
from src.data_models.data_models import StyleExtractorImplementation
from src.style_extraction.style_extractor import StyleExtractor


def build_style_extractor(
    style_extractor_implementation: StyleExtractorImplementation | None = None,
) -> StyleExtractor:
    """
    Construct a ready-made style extractor from pre-configured implementations.

    Args:
        style_extractor_implementation (StyleExtractorImplementation | None): Choice of
            a pre-configured implementation. If None, defaults to the value from the
            project-wide configuration.

    Returns:
        StyleExtractor: An initialised instance of the style extractor.
    """
    # Local imports for lazy loading prevent importing heavy implementation
    # modules when they are not requested.
    # pylint: disable=import-outside-toplevel
    from src.style_extraction.spacy_style_extractor import SpacyStyleExtractor
    # pylint: enable=import-outside-toplevel

    style_extractor_implementation = (
        style_extractor_implementation or config.style_extractor
    )

    match style_extractor_implementation:
        case StyleExtractorImplementation.SPACY:
            return SpacyStyleExtractor()

"""Module with the SpaCy-based implementation of a style extractor."""

import numpy as np
import pytest

from src.style_extraction.spacy_style_extractor import SpacyStyleExtractor
from src.style_extraction.style_extractor import StyleExtractor
from src.utils.errors import NotPreparedError


@pytest.fixture
def spacy_extractor() -> StyleExtractor:
    """Fixture providing an unprepared SpaCy-based style extractor."""
    return SpacyStyleExtractor()


@pytest.fixture
def sentences() -> list[str]:
    """Fixture providing a list of several sentences."""
    return [
        "Hello, world!",
        "Contradictor is such a cool project!",
        "Life was born on Earth.",
        (
            "The capabilities of a processing pipeline always depend on the "
            "components, their models and how they were trained. "
        ),
        (
            "The projects, which I self-host, include paperless-ngx, forgejo, "
            "nginx reverse proxy manager, audiobookshelf; many more are coming."
        ),
    ]


@pytest.mark.asyncio
async def test_unprepared_extraction_raises_error(
    spacy_extractor: StyleExtractor, sentences: list[str]
) -> None:
    """Test if an unprepared extractor raises an exception."""
    with pytest.raises(NotPreparedError):
        await spacy_extractor.extract(texts=sentences)


@pytest.mark.asyncio
async def test_reproducible_extraction(
    spacy_extractor: StyleExtractor, sentences: list[str]
) -> None:
    """Test if running the the same instance twice gives identical results."""
    await spacy_extractor.prepare()

    representations_1 = await spacy_extractor.extract(sentences)
    representations_2 = await spacy_extractor.extract(sentences)

    for representation_1, representation_2 in zip(
        representations_1, representations_2, strict=True
    ):
        assert np.allclose(representation_1, representation_2)


@pytest.mark.asyncio
async def test_reproducible_extraction_with_different_instances(
    sentences: list[str],
) -> None:
    """Test if running the the different instances gives identical results."""
    first_extractor = SpacyStyleExtractor()
    await first_extractor.prepare()

    second_extractor = SpacyStyleExtractor()
    await second_extractor.prepare()

    representations_1 = await first_extractor.extract(sentences)
    representations_2 = await second_extractor.extract(sentences)

    for representation_1, representation_2 in zip(
        representations_1, representations_2, strict=True
    ):
        assert np.allclose(representation_1, representation_2)

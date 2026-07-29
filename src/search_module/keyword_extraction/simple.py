"""Simple keyword extraction."""

from collections import Counter
from typing import override

from src.data_models.abstract.tokeniser import Tokeniser
from src.search_module.config import KeywordConfig
from src.search_module.interfaces import KeywordExtractor
from src.search_module.models import SearchQuery


class SimpleKeywordExtractor(KeywordExtractor):
    """Keyword extractor based on word frequency."""

    def __init__(
        self,
        config: KeywordConfig,
        tokeniser: Tokeniser,
    ) -> None:
        """Initialize keyword extractor with configuration and a tokeniser."""
        self.config = config
        self.tokeniser = tokeniser

    @override
    def extract(
        self,
        text: str,
    ) -> SearchQuery:
        words = self.tokeniser.tokenise(text)
        keywords = [
            word for word, _ in Counter(words).most_common(self.config.max_keywords)
        ]

        return SearchQuery(
            original_text=text,
            keywords=tuple(keywords),
            normalized=" ".join(keywords),
        )

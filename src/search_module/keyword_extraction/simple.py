"""Simple keyword extraction."""

from collections import Counter
from typing import override

from src.data_models.abstract.configuration import ConfigurationValue
from src.search_module.config import KeywordConfig
from src.search_module.interfaces import KeywordExtractor
from src.search_module.models import SearchQuery
from src.utils.tokenisers.tokeniser import Tokeniser


class SimpleKeywordExtractor(KeywordExtractor):
    """Keyword extractor based on word frequency."""

    def __init__(
        self,
        config: KeywordConfig,
        tokeniser: Tokeniser,
    ) -> None:
        """
        Initialise the keyword extractor with configuration and a tokeniser.

        Args:
            config (KeywordConfig): Configuration for keyword extraction.
            tokeniser (Tokeniser): Tokeniser used to split text into words.
        """
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

    @override
    def get_configuration(self) -> dict[str, ConfigurationValue]:
        return {
            "max_keywords": self.config.max_keywords,
        }

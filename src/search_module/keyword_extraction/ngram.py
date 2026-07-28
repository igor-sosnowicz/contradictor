"""N-gram keyword extractor implementation."""

import logging
from collections import Counter

from src.data_models.abstract.tokeniser import Tokeniser
from src.search_module.config import KeywordConfig
from src.search_module.interfaces import KeywordExtractor
from src.search_module.models import SearchQuery

logger = logging.getLogger(__name__)


class NGramKeywordExtractor(KeywordExtractor):
    """Keyword extractor using n-gram frequency analysis."""

    def __init__(
        self,
        config: KeywordConfig,
        tokeniser: Tokeniser,
    ) -> None:
        """Initialize the extractor with configuration and a tokeniser."""
        self.config = config
        self.tokeniser = tokeniser

    def extract(
        self,
        text: str,
    ) -> SearchQuery:
        """Extract keywords from text using n-gram frequency analysis."""
        tokens = self.tokeniser.tokenise(text)
        candidates = self._create_ngrams(tokens)
        keywords = [
            phrase
            for phrase, _ in Counter(candidates).most_common(self.config.max_keywords)
        ]

        return SearchQuery(
            original_text=text,
            keywords=tuple(keywords),
            normalized=" ".join(keywords),
        )

    def _create_ngrams(
        self,
        tokens: list[str],
    ) -> list[str]:
        """Generate n-grams based on configured min and max sizes."""
        result: list[str] = []
        ngram_range = range(self.config.min_ngram_size, self.config.max_ngram_size + 1)

        for size in ngram_range:
            result.extend(
                " ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)
            )

        return result

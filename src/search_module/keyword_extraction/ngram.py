"""N-gram keyword extraction."""

import re
from collections import Counter

from src.search_module.config import KeywordConfig
from src.search_module.interfaces import KeywordExtractor
from src.search_module.models import SearchQuery


class NGramKeywordExtractor(KeywordExtractor):
    """
    Keyword extractor using stop words
    and n-gram frequency analysis.
    """

    def __init__(
        self,
        config: KeywordConfig,
    ) -> None:
        """Initialize n-gram keyword extractor with configuration."""
        self.config = config

    def extract(
        self,
        text: str,
    ) -> SearchQuery:
        """Extract keywords from text using n-gram frequency analysis."""
        tokens = self._tokenize(text)

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

    def _tokenize(
        self,
        text: str,
    ) -> list[str]:

        words = re.findall(
            rf"\b[a-zA-Z]{{{self.config.min_word_length},}}\b",
            text.lower(),
        )

        return [word for word in words if word not in self.config.stop_words]

    def _create_ngrams(
        self,
        tokens: list[str],
    ) -> list[str]:

        result: list[str] = []

        for size in self.config.ngram_sizes:
            result.extend(
                " ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)
            )

        return result

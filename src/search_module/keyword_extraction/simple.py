"""Simple keyword extraction."""

from collections import Counter

from src.search_module.interfaces import KeywordExtractor
from src.search_module.models import SearchQuery
from src.search_module.utils.utils import tokenize


class SimpleKeywordExtractor(KeywordExtractor):
    """Keyword extractor based on word frequency."""

    def __init__(
        self,
        stop_words: set[str] | None = None,
        max_keywords: int = 5,
    ) -> None:
        """Initialize keyword extractor with stop words and keyword limit."""
        self.stop_words = stop_words or set()
        self.max_keywords = max_keywords

    def extract(
        self,
        text: str,
    ) -> SearchQuery:
        """Keyword extractor based on heuristic rules."""
        words = self._tokenize(text)

        keywords = [word for word, _ in Counter(words).most_common(self.max_keywords)]

        return SearchQuery(
            original_text=text,
            keywords=tuple(keywords),
            normalized=" ".join(keywords),
        )

    def _tokenize(
        self,
        text: str,
    ) -> list[str]:
        return tokenize(
            text,
            min_word_length=3,
            stop_words=self.stop_words,
        )

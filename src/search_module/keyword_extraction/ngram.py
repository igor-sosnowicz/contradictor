"""N-gram keyword extractor implementation."""

from collections import Counter

from src.search_module.config import KeywordConfig
from src.search_module.interfaces import KeywordExtractor
from src.search_module.models import SearchQuery
from src.utils.tokenisers.tokeniser import Tokeniser


class NGramKeywordExtractor(KeywordExtractor):
    """Keyword extractor using n-gram frequency analysis."""

    def __init__(
        self,
        config: KeywordConfig,
        tokeniser: Tokeniser,
    ) -> None:
        """
        Initialize the extractor with configuration and a tokeniser.

        Args:
            config (KeywordConfig): Configuration for keyword extraction.
            tokeniser (Tokeniser): Tokeniser used to split text into tokens.

        Returns:
            None
        """
        self.config = config
        self.tokeniser = tokeniser

    def extract(
        self,
        text: str,
    ) -> SearchQuery:
        """
        Extract keywords from text using n-gram frequency analysis.

        Args:
            text (str): The input text from which keywords should be extracted.

        Returns:
            SearchQuery: A normalized search query containing the extracted
                keywords.
        """
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
        """
        Generate n-grams based on configured minimum and maximum sizes.

        Args:
            tokens (list[str]): A sequence of tokens from which to generate
                n-grams.

        Returns:
            list[str]: A list of generated n-gram phrases.
        """
        result: list[str] = []
        ngram_range = range(self.config.min_ngram_size, self.config.max_ngram_size + 1)

        for size in ngram_range:
            result.extend(
                " ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)
            )

        return result

"""KeyBERT based keyword extraction."""

from collections.abc import Generator
from typing import override

from keybert import KeyBERT

from src.search_module.config import KeywordConfig
from src.search_module.interfaces import KeywordExtractor
from src.search_module.models import SearchQuery


class KeyBERTKeywordExtractor(KeywordExtractor):
    """Keyword extractor using KeyBERT embeddings."""

    def __init__(
        self,
        config: KeywordConfig,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        """
        Initialize the keyword extractor with a KeyBERT model.

        Args:
            config (KeywordConfig): Configuration for keyword extraction.
            model_name (str): Name of the sentence-transformer model used by
                KeyBERT. Defaults to "all-MiniLM-L6-v2".

        Returns:
            None
        """
        self.config = config
        self.model = KeyBERT(model=model_name)

    @override
    def extract(
        self,
        text: str,
    ) -> SearchQuery:
        keywords = self.model.extract_keywords(
            text,
            keyphrase_ngram_range=(
                self.config.min_ngram_size,
                self.config.max_ngram_size,
            ),
            stop_words=list(self.config.stop_words),
            top_n=self.config.max_keywords,
        )
        phrases: list[str] = list(_chain(keywords))

        return SearchQuery(
            original_text=text,
            keywords=tuple(phrases),
            normalized=" ".join(phrases),
        )


def _chain(
    scored: list[tuple[str, float]] | list[list[tuple[str, float]]],
) -> Generator[str]:
    """
    Flatten List[Tuple[str, float]] | List[List[Tuple[str, float]]]
    and yield only the str elements from tuples.
    """
    if isinstance(scored, list):
        for item in scored:
            if isinstance(item, list):
                yield from _chain(item)
            elif isinstance(item, tuple) and len(item) >= 1:
                yield item[0]
    elif isinstance(scored, tuple) and len(scored) >= 1:
        yield scored[0]

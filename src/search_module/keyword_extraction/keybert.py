"""KeyBERT based keyword extraction."""

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
        """Initialize keyword extractor with KeyBERT model configuration."""
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

        phrases = [keyword for keyword, _ in keywords]

        return SearchQuery(
            original_text=text,
            keywords=tuple(phrases),
            normalized=" ".join(phrases),
        )

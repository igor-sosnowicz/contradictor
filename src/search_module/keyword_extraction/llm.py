"""LLM based keyword extraction."""

from src.search_module.interfaces import KeywordExtractor, LLMClient
from src.search_module.models import SearchQuery


class LLMKeywordExtractor(KeywordExtractor):
    """
    Keyword extractor using an external LLM.

    The LLM client is injected to keep
    provider independence.
    """

    def __init__(
        self,
        client: LLMClient,
        max_keywords: int = 5,
    ) -> None:
        """Initialize LLM keyword extractor."""
        self.client = client
        self.max_keywords = max_keywords

    def extract(
        self,
        text: str,
    ) -> SearchQuery:
        """
        Extract keywords from text using an LLM.

        Args:
            text: Text to analyze.

        Returns:
            SearchQuery containing extracted keywords and normalized query.
        """
        keywords = self.client.extract_keywords(
            text,
            limit=self.max_keywords,
        )

        return SearchQuery(
            original_text=text,
            keywords=tuple(keywords),
            normalized=" ".join(keywords),
        )

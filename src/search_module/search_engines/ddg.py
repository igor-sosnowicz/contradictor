"""DuckDuckGo search engine implementation."""

from typing import override

from ddgs import DDGS
from loguru import logger

from src.search_module.config import SearchConfig
from src.search_module.interfaces import SearchEngine
from src.search_module.models import SearchQuery, SearchResult
from src.utils.errors import ContradictorError


class DDGSearchEngine(SearchEngine):
    """
    Initialize the DuckDuckGo search engine with configuration.

    Args:
        config (SearchConfig): Configuration for the search engine.

    Returns:
        None
    """

    def __init__(
        self,
        config: SearchConfig,
    ) -> None:
        """Initialize DuckDuckGo search engine with configuration."""
        self.config = config

    @override
    def search(
        self,
        query: SearchQuery,
        limit: int = 10,
    ) -> list[SearchResult]:
        logger.info(
            "DuckDuckGo search: {}",
            query.normalized,
        )
        if not query.normalized.strip():
            return []

        max_results = limit if limit is not None else self.config.max_results
        results: list[SearchResult] = []
        try:
            with DDGS() as ddgs:
                search_results = ddgs.text(
                    query.normalized,
                    region=self.config.region,
                    max_results=max_results,
                )
                logger.info("DuckDuckGo returned results")
                results = [
                    SearchResult(
                        url=item["href"],
                        title=item.get("title", ""),
                        snippet=item.get("body", ""),
                    )
                    for item in search_results
                ]
        except ContradictorError:
            logger.exception(
                "DuckDuckGo search failed: {}",
                query.normalized,
            )
            return []
        return results

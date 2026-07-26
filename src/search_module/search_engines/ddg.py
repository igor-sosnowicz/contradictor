"""DuckDuckGo search engine implementation."""

import logging

from ddgs import DDGS

from src.search_module.config import SearchConfig
from src.search_module.interfaces import SearchEngine
from src.search_module.models import SearchQuery, SearchResult
from src.utils.errors import ContradictorError

logger = logging.getLogger(__name__)


class DDGSearchEngine(SearchEngine):
    """
    Search engine implementation using DuckDuckGo.

    Responsible only for searching.
    """

    def __init__(
        self,
        config: SearchConfig,
    ) -> None:
        """Initialize DuckDuckGo search engine with configuration."""
        self.config = config

    def search(
        self,
        query: SearchQuery,
        limit: int | None = None,
    ) -> list[SearchResult]:
        """Search DuckDuckGo and return matching results."""
        logger.info(
            "DuckDuckGo search: %s",
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
                "DuckDuckGo search failed: %s",
                query.normalized,
            )
            return []

        return results

"""Search pipeline orchestration."""

import asyncio
from dataclasses import dataclass
from threading import BoundedSemaphore

import httpx
from loguru import logger

from src.search_module.config import ConcurrencyConfig
from src.search_module.interfaces import (
    CacheBackend,
    Cleaner,
    Downloader,
    KeywordExtractor,
    SearchEngine,
)
from src.search_module.models import Document, SearchResult
from src.search_module.utils.cache_keys import create_cache_key
from src.utils.errors import ContradictorError


@dataclass
class SearchPipelineDependencies:
    """Dependencies required by SearchPipeline."""

    keyword_extractor: KeywordExtractor
    search_engine: SearchEngine
    downloader: Downloader
    cleaner: Cleaner
    cache: CacheBackend
    concurrency: ConcurrencyConfig


class SearchPipeline:
    """
    Orchestrates the complete document search workflow.

    Coordinates keyword extraction, searching, downloading,
    cleaning, and caching.
    """

    def __init__(
        self,
        dependencies: SearchPipelineDependencies,
    ) -> None:
        """Initialise search pipeline dependencies."""
        self._search_semaphore = BoundedSemaphore(
            dependencies.concurrency.max_concurrent_searches
        )
        self._download_semaphore = asyncio.Semaphore(
            dependencies.concurrency.max_workers
        )

        self.keyword_extractor = dependencies.keyword_extractor
        self.search_engine = dependencies.search_engine
        self.downloader = dependencies.downloader
        self.cleaner = dependencies.cleaner
        self.cache = dependencies.cache

    async def search(
        self,
        text: str,
    ) -> list[Document]:
        """
        Run the search pipeline and return cleaned documents.

        Args:
            text: User input from which search keywords are extracted.

        Returns:
            Cleaned documents retrieved from search results.
        """
        logger.info("Starting search pipeline")
        query = self.keyword_extractor.extract(text)
        logger.info("Extracted keywords: {}", query.keywords)
        cache_key = create_cache_key(query.normalized)
        logger.debug("Cache key: {}", cache_key)
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.info("Cache hit: {}", cache_key)
            return cached
        logger.info("Searching: {}", query.normalized)
        with self._search_semaphore:
            results = self.search_engine.search(query)
        logger.info("Search returned {} results", len(results))
        downloaded_results = await asyncio.gather(
            *(self._download_result(result) for result in results)
        )
        documents: list[Document] = []

        for url, html in downloaded_results:
            if not html:
                continue
            try:
                document = self.cleaner.clean(html, url)
                if not document.text:
                    logger.warning(
                        "Empty document after cleaning: {}",
                        url,
                    )
                    continue
                documents.append(document)
                logger.info("Accepted document: {}", url)
            except (ContradictorError, ValueError):
                logger.exception(
                    "Failed processing document: {}",
                    url,
                )
        logger.info(
            "Cleaning finished. Documents: {}",
            len(documents),
        )
        self.cache.set(cache_key, documents)
        logger.info("Saved results to cache")
        return documents

    async def _download_result(
        self,
        result: SearchResult,
    ) -> tuple[str, str]:
        """Download HTML for a single search result."""
        logger.debug("Processing URL: {}", result.url)
        async with self._download_semaphore:
            try:
                html = await self.downloader.download(result.url)
                logger.debug(
                    "Downloaded {} characters from {}",
                    len(html),
                    result.url,
                )
            except (
                httpx.HTTPError,
                httpx.InvalidURL,
                ContradictorError,
                ValueError,
            ):
                logger.exception(
                    "Failed downloading URL: {}",
                    result.url,
                )
                return result.url, ""

            return result.url, html

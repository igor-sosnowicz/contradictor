"""Search pipeline orchestration."""

from loguru import logger

from src.search_module.interfaces import (
    CacheBackend,
    Cleaner,
    Downloader,
    KeywordExtractor,
    SearchEngine,
)
from src.search_module.models import Document
from src.search_module.utils.cache_keys import create_cache_key
from src.utils.errors import ContradictorError


class SearchPipeline:
    """
    Orchestrates the complete document search workflow.

    Coordinates keyword extraction, searching, downloading,
    cleaning, and caching. This class does not implement the
    individual processing steps; it only controls their execution.
    """

    def __init__(
        self,
        keyword_extractor: KeywordExtractor,
        search_engine: SearchEngine,
        downloader: Downloader,
        cleaner: Cleaner,
        cache: CacheBackend,
    ) -> None:
        """
        Initialize search pipeline dependencies.

        Args:
            keyword_extractor (KeywordExtractor): Component responsible for
                extracting search keywords from user input.
            search_engine (SearchEngine): Component responsible for executing
                searches.
            downloader (Downloader): Component responsible for downloading
                web page content.
            cleaner (Cleaner): Component responsible for extracting readable
                content from downloaded pages.
            cache (CacheBackend): Storage backend for cached search results.

        Returns:
            None: Initializes pipeline components.
        """
        self.keyword_extractor = keyword_extractor
        self.search_engine = search_engine
        self.downloader = downloader
        self.cleaner = cleaner
        self.cache = cache

    def search(
        self,
        text: str,
    ) -> list[Document]:
        """Run search pipeline and return cleaned documents."""
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
        results = self.search_engine.search(query)
        logger.info("Search returned {} results", len(results))

        documents: list[Document] = []
        for result in results:
            logger.debug("Processing URL: {}", result.url)
            try:
                html = self.downloader.download(result.url)
                logger.debug("Downloaded {} characters from {}", len(html), result.url)

                document = self.cleaner.clean(html, result.url)
                if not document.text:
                    logger.warning("Empty document after cleaning: {}", result.url)
                    continue
                documents.append(document)
                logger.info("Accepted document: {}", result.url)

            except ContradictorError:
                logger.exception("Failed processing {}", result.url)

        logger.info("Cleaning finished. Documents: {}", len(documents))
        self.cache.set(cache_key, documents)
        logger.info("Saved results to cache")

        return documents

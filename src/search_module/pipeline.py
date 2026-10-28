"""Search pipeline orchestration."""

from concurrent.futures import ThreadPoolExecutor

from loguru import logger

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


class SearchPipeline:
    """
    Orchestrates the complete document search workflow.

    Coordinates keyword extraction, searching, downloading,
    cleaning, and caching. This class does not implement the
    individual processing steps; it only controls their execution.
    """

    # The optional max_workers parameter preserves backward compatibility.
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(  # noqa: PLR0913
        self,
        keyword_extractor: KeywordExtractor,
        search_engine: SearchEngine,
        downloader: Downloader,
        cleaner: Cleaner,
        cache: CacheBackend,
        max_workers: int = 5,
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
            max_workers (int): Maximum number of concurrent download
                workers. Defaults to 5.

        Returns:
            None: Initializes pipeline components.
        """
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")

        self.keyword_extractor = keyword_extractor
        self.search_engine = search_engine
        self.downloader = downloader
        self.cleaner = cleaner
        self.cache = cache
        self.max_workers = max_workers

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

        with ThreadPoolExecutor(
            max_workers=self.max_workers,
        ) as executor:
            downloaded_results = list(executor.map(self._download_result, results))

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

            except ContradictorError:
                logger.exception("Failed processing {}", url)

        logger.info("Cleaning finished. Documents: {}", len(documents))
        self.cache.set(cache_key, documents)
        logger.info("Saved results to cache")

        return documents

    def _download_result(self, result: SearchResult) -> tuple[str, str]:
        """Download HTML for a single search result."""
        logger.debug("Processing URL: {}", result.url)

        try:
            html = self.downloader.download(result.url)
            logger.debug(
                "Downloaded {} characters from {}",
                len(html),
                result.url,
            )
        except ContradictorError:
            logger.exception("Failed processing {}", result.url)
            html = ""

        return result.url, html

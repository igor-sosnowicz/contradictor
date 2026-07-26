"""Search pipeline orchestration."""

import logging

from src.search_module.interfaces import (
    CacheBackend,
    Cleaner,
    Downloader,
    KeywordExtractor,
    SearchEngine,
)
from src.search_module.models import (
    Document,
)
from src.search_module.utils.cache_keys import create_cache_key
from src.utils.errors import ContradictorError

logger = logging.getLogger(__name__)


class SearchPipeline:
    """
    Complete search pipeline.
    Responsible only for orchestration.
    """

    def __init__(
        self,
        keyword_extractor: KeywordExtractor,
        search_engine: SearchEngine,
        downloader: Downloader,
        cleaner: Cleaner,
        cache: CacheBackend,
    ) -> None:
        """Initialize search pipeline with required components."""
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

        logger.info(
            "Extracted keywords: %s",
            query.keywords,
        )

        cache_key = create_cache_key(query.normalized)

        logger.debug(
            "Cache key: %s",
            cache_key,
        )

        cached = self.cache.get(cache_key)

        if cached is not None:
            logger.info(
                "Cache hit: %s",
                cache_key,
            )

            return cached

        logger.info(
            "Searching: %s",
            query.normalized,
        )

        results = self.search_engine.search(query)

        logger.info(
            "Search returned %d results",
            len(results),
        )

        documents: list[Document] = []

        for result in results:
            logger.debug(
                "Processing URL: %s",
                result.url,
            )

            try:
                html = self.downloader.download(result.url)

                logger.debug(
                    "Downloaded %d characters from %s",
                    len(html),
                    result.url,
                )

                document = self.cleaner.clean(
                    html,
                    result.url,
                )

                if not document.text:
                    logger.warning(
                        "Empty document after cleaning: %s",
                        result.url,
                    )

                    continue

                if document.text:
                    documents.append(document)

                logger.info(
                    "Accepted document: %s",
                    result.url,
                )

            except ContradictorError:
                logger.exception(
                    "Failed processing %s",
                    result.url,
                )

        logger.info(
            "Cleaning finished. Documents: %d",
            len(documents),
        )

        self.cache.set(
            cache_key,
            documents,
        )

        logger.info("Saved results to cache")

        return documents

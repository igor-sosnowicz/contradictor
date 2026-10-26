"""Module with a search pipeline factory function."""

from src.benchmark.golden_dataset import GoldenDatasetLoader
from src.configuration import config
from src.data_models.data_models import SearchPipelineImplementation
from src.search_module.cleaners import NullCleaner, ReadabilityCleaner
from src.search_module.config import (
    CacheConfig,
    CleaningConfig,
    DownloadConfig,
    KeywordConfig,
    SearchConfig,
)
from src.search_module.downloaders.local_downloader import LocalDownloader
from src.search_module.pipeline import SearchPipeline
from src.search_module.search_engines.local_dataset import LocalDatasetSearchEngine


def build_search_pipeline(
    search_implementation: SearchPipelineImplementation | None = None,
) -> SearchPipeline:
    """
    Construct a ready-made search pipeline from pre-configured implementations.

    Args:
        search_implementation (SearchPipelineImplementation): Choice of a
            pre-configured implementation.

    Returns:
        SearchPipeline: An initialised instance of the search pipeline.

    Raises:
        NotImplementedError: Raised if not implemented variant was requested.
    """
    # Local imports for lazy loading prevent importing heavy implementation
    # modules when they are not requested.
    # pylint: disable=import-outside-toplevel
    from src.search_module.cache import DiskCacheBackend
    from src.search_module.downloaders.request_downloader import RequestsDownloader
    from src.search_module.keyword_extraction.keybert import KeyBERTKeywordExtractor
    from src.search_module.search_engines.ddg import DDGSearchEngine
    # pylint: enable=import-outside-toplevel

    search_implementation = search_implementation or config.search_pipeline

    match search_implementation:
        case SearchPipelineImplementation.REMOTE:
            return SearchPipeline(
                keyword_extractor=KeyBERTKeywordExtractor(config=KeywordConfig()),
                search_engine=DDGSearchEngine(config=SearchConfig()),
                cleaner=ReadabilityCleaner(config=CleaningConfig()),
                downloader=RequestsDownloader(config=DownloadConfig()),
                cache=DiskCacheBackend(config=CacheConfig()),
            )

        case SearchPipelineImplementation.LOCAL_DATASET:
            golden_dataset = GoldenDatasetLoader().load()
            return SearchPipeline(
                keyword_extractor=KeyBERTKeywordExtractor(config=KeywordConfig()),
                search_engine=LocalDatasetSearchEngine(dataset=golden_dataset),
                cleaner=NullCleaner(),
                downloader=LocalDownloader(dataset=golden_dataset),
                cache=DiskCacheBackend(config=CacheConfig()),
            )

"""End-to-End (E2E) test for the search module pipeline without any fakes."""

import re
from pathlib import Path

from src.search_module.cache import DiskCacheBackend
from src.search_module.cleaner import ReadabilityCleaner
from src.search_module.config import (
    CacheConfig,
    CleaningConfig,
    SearchConfig,
    SearchModuleConfig,
)
from src.search_module.downloader import RequestsDownloader
from src.search_module.keyword_extraction.ngram import NGramKeywordExtractor
from src.search_module.pipeline import SearchPipeline
from src.search_module.search_engines.ddg import DDGSearchEngine
from src.utils.tokenisers.tokeniser import Tokeniser


class BenchmarkTokeniser(Tokeniser):
    """Tokeniser used for producing real tokens during integration tests."""

    def encode(self, text: list[str]) -> list[list[int]]:
        """
        Encode a list of text strings into lists of integer token IDs.

        Args:
            text (list[str]): A list of strings to be tokenised.

        Returns:
            list[list[int]]: A list containing lists of integer token IDs
            for each input string.
        """
        return [[ord(char) for char in item] for item in text]

    def decode(self, tokens: list[list[int]]) -> list[str]:
        """
        Decode lists of token IDs back into a list of text strings.

        Args:
            tokens (list[list[int]]): A list of token ID lists to decode.

        Returns:
            list[str]: A list of decoded text strings.
        """
        return ["".join(chr(token) for token in item) for item in tokens]

    def tokenise(self, text: str) -> list[str]:
        """
        Tokenise a text string into individual words using a regular expression.

        Extracts lowercase words that are at least two characters long and
        consist exclusively of Latin letters.

        Args:
            text (str): The input text string to tokenise.

        Returns:
            list[str]: A list of extracted word tokens.
        """
        return re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())


def test_pipeline_complete_e2e_flow(tmp_path: Path) -> None:
    """
    Execute a complete end-to-end integration test for the search pipeline.

    Validates that the entire sequence—from regular expression keyword
    extraction to executing a live duckduckgo web search, downloading content,
    cleaning HTML tags, and reusing cached data—functions correctly together.
    """
    config = SearchModuleConfig(
        cache=CacheConfig(
            enabled=True,
        ),
        search=SearchConfig(
            max_results=2,
        ),
        cleaning=CleaningConfig(
            min_text_length=100,
        ),
    )

    tokeniser = BenchmarkTokeniser()
    keyword_extractor = NGramKeywordExtractor(
        config=config.keyword,
        tokeniser=tokeniser,
    )
    search_engine = DDGSearchEngine(config=config.search)
    downloader = RequestsDownloader(config=config.download)
    cleaner = ReadabilityCleaner(config=config.cleaning)

    with DiskCacheBackend(config=config.cache, directory=tmp_path) as cache:
        pipeline = SearchPipeline(
            keyword_extractor=keyword_extractor,
            search_engine=search_engine,
            downloader=downloader,
            cleaner=cleaner,
            cache=cache,
        )

        search_phrase = "quantum computing breakthrough 2026"
        documents = pipeline.search(search_phrase)

        assert isinstance(documents, list)
        assert len(documents) > 0, "Pipeline returned no documents from live internet."

        first_doc = documents[0]
        assert first_doc.url.startswith("http")
        assert len(first_doc.title) > 0
        assert len(first_doc.text) >= config.cleaning.min_text_length

        assert "<html" not in first_doc.text
        assert "<body>" not in first_doc.text

        second_pipeline = SearchPipeline(
            keyword_extractor=keyword_extractor,
            search_engine=search_engine,
            downloader=downloader,
            cleaner=cleaner,
            cache=cache,
        )

        cached_documents = second_pipeline.search(search_phrase)

        assert len(cached_documents) == len(documents)
        assert cached_documents[0].text == first_doc.text

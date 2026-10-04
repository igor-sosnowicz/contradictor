"""Real Integration test for the search module pipeline without any fakes."""

import re
from pathlib import Path

from src.search_module.cache import DiskCacheBackend
from src.search_module.cleaner import ReadabilityCleaner
from src.search_module.config import SearchModuleConfig
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


def test_pipeline_ngram_and_readability_real_integration(
    tmp_path: Path,
) -> None:
    """Perform an integration test of the search pipeline with internet components."""
    config = SearchModuleConfig()
    config.cache.enabled = True
    config.search.max_results = 2
    config.cleaning.min_text_length = 150
    config.keyword.min_ngram_size = 2
    config.keyword.max_ngram_size = 3

    tokeniser = BenchmarkTokeniser()
    keyword_extractor = NGramKeywordExtractor(
        config=config.keyword,
        tokeniser=tokeniser,
    )
    search_engine = DDGSearchEngine(config=config.search)
    downloader = RequestsDownloader(config=config.download)
    readability_cleaner = ReadabilityCleaner(config=config.cleaning)

    with DiskCacheBackend(config=config.cache, directory=tmp_path) as cache:
        pipeline = SearchPipeline(
            keyword_extractor=keyword_extractor,
            search_engine=search_engine,
            downloader=downloader,
            cleaner=readability_cleaner,
            cache=cache,
        )

        search_phrase = "quantum computing breakthrough 2026"

        documents = pipeline.search(search_phrase)

        assert isinstance(documents, list)
        assert len(documents) > 0, (
            "Pipeline returned no documents from live internet search."
        )

        document = documents[0]

        assert document.url.startswith("http")
        assert len(document.title) > 0
        assert len(document.text) >= config.cleaning.min_text_length
        assert "<html" not in document.text
        assert "<body>" not in document.text

        second_pipeline = SearchPipeline(
            keyword_extractor=keyword_extractor,
            search_engine=search_engine,
            downloader=downloader,
            cleaner=readability_cleaner,
            cache=cache,
        )

        cached_documents = second_pipeline.search(search_phrase)

        assert len(cached_documents) == len(documents)
        assert cached_documents[0].text == document.text

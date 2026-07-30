"""Tests for the pipeline of the search module."""

from src.search_module.models import (
    Document,
    SearchQuery,
    SearchResult,
)
from src.search_module.pipeline import SearchPipeline
from src.search_module.utils.cache_keys import create_cache_key


class FakeKeywordExtractor:
    """Fake keyword extractor for pipeline tests."""

    def extract(
        self,
        text: str,
    ) -> SearchQuery:
        """Return predefined search query."""
        return SearchQuery(
            original_text=text,
            keywords=("cats",),
            normalized="cats",
        )


class FakeSearchEngine:
    """Fake search engine for pipeline tests."""

    def __init__(self) -> None:
        """Initialize fake search engine state."""
        self.called = False

    def search(
        self,
        query: SearchQuery,
    ) -> list[SearchResult]:
        """Return predefined search results."""
        self.called = True

        return [
            SearchResult(
                url="https://example.com",
                title="Cats",
            )
        ]


class FakeDownloader:
    """Fake downloader for pipeline tests."""

    def download(
        self,
        url: str,
    ) -> str:
        """Return fake HTML content."""
        return (
            "<article>"
            "Cats are great pets. "
            "Cats are intelligent animals. "
            "Cats sleep a lot."
            "</article>"
        )


class FakeCleaner:
    """Fake cleaner for pipeline tests."""

    def clean(
        self,
        html: str,
        url: str,
    ) -> Document:
        """Return predefined cleaned document."""
        return Document(
            url=url,
            text="Cats are great pets.",
        )


class FakeCache:
    """Fake cache backend for pipeline tests."""

    def __init__(self) -> None:
        """Initialize empty cache."""
        self.storage: dict[str, list[Document]] = {}

    def get(
        self,
        key: str,
    ) -> list[Document] | None:
        """Return cached value by key."""
        return self.storage.get(key)

    def set(
        self,
        key: str,
        value: list[Document],
    ) -> None:
        """Store value under cache key."""
        self.storage[key] = value

    def clear(self) -> None:
        """Clear cached values."""
        self.storage.clear()


def test_pipeline_search_returns_documents() -> None:
    """Verify that pipeline returns cleaned documents."""
    pipeline = SearchPipeline(
        FakeKeywordExtractor(),
        FakeSearchEngine(),
        FakeDownloader(),
        FakeCleaner(),
        FakeCache(),
    )

    documents = pipeline.search("cats")

    assert len(documents) == 1
    assert documents[0].url == "https://example.com"


def test_pipeline_uses_cache() -> None:
    """Verify that cached results skip searching."""
    cache = FakeCache()

    cached_documents = [
        Document(
            url="https://example.com",
            text="Cached document",
        )
    ]

    cache.set(
        create_cache_key("cats"),
        cached_documents,
    )

    engine = FakeSearchEngine()

    pipeline = SearchPipeline(
        FakeKeywordExtractor(),
        engine,
        FakeDownloader(),
        FakeCleaner(),
        cache,
    )

    result = pipeline.search("cats")

    assert result == cached_documents
    assert engine.called is False

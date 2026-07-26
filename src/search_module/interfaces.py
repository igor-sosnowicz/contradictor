"""Module with search module interfaces."""

from abc import ABC, abstractmethod
from typing import Protocol

from src.search_module.models import Document, SearchQuery, SearchResult


class KeywordExtractor(ABC):
    """Extract keywords from an input text."""

    @abstractmethod
    def extract(self, text: str) -> SearchQuery:
        """Extract keywords and return a normalized search query."""
        raise NotImplementedError


class SearchEngine(ABC):
    """Search engine interface."""

    @abstractmethod
    def search(
        self,
        query: SearchQuery,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Return search results."""
        raise NotImplementedError


class Downloader(ABC):
    """Downloads web pages."""

    @abstractmethod
    def download(self, url: str) -> str:
        """Download raw HTML."""
        raise NotImplementedError


class Cleaner(ABC):
    """Converts HTML into clean text."""

    @abstractmethod
    def clean(
        self,
        html: str,
        url: str,
    ) -> Document:
        """Extract clean text from HTML content and return a document."""
        raise NotImplementedError


class CacheBackend(ABC):
    """Persistent cache interface."""

    @abstractmethod
    def get(
        self,
        key: str,
    ) -> list[Document] | None:
        """Retrieve cached documents by key."""
        raise NotImplementedError

    @abstractmethod
    def set(
        self,
        key: str,
        value: list[Document],
    ) -> None:
        """Store documents in cache under a key."""
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Remove all cached entries."""
        raise NotImplementedError


class LLMClient(Protocol):
    """Interface for LLM keyword extraction clients."""

    def extract_keywords(
        self,
        text: str,
        limit: int,
    ) -> list[str]:
        """Extract keywords from text."""
        ...

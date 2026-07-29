"""Module with search module interfaces."""

from abc import ABC, abstractmethod
from typing import Protocol

from src.search_module.models import Document, SearchQuery, SearchResult


class KeywordExtractor(ABC):
    """Interface for extracting search keywords from unstructured text."""

    @abstractmethod
    def extract(self, text: str) -> SearchQuery:
        """
        Extract keywords and return a normalized search query.

        Args:
            text: The raw input text from which keywords should be extracted.

        Returns:
            A SearchQuery object containing the extracted and normalized keywords.
        """


class SearchEngine(ABC):
    """Interface for executing queries against external search engines."""

    @abstractmethod
    def search(
        self,
        query: SearchQuery,
        limit: int = 10,
    ) -> list[SearchResult]:
        """
        Execute a search query and fetch search results.

        Args:
            query: The normalized search query object.
            limit: The maximum number of results to return. Defaults to 10.

        Returns:
            A list of SearchResult objects matching the query.
        """


class Downloader(ABC):
    """Interface for downloading raw web page content."""

    @abstractmethod
    def download(self, url: str) -> str:
        """
        Download raw HTML content from the specified URL.

        Args:
            url: The absolute HTTP/HTTPS URL of the target web page.

        Returns:
            The raw HTML content as a string.
        """


class Cleaner(ABC):
    """Interface for converting raw HTML into structured, readable documents."""

    @abstractmethod
    def clean(
        self,
        html: str,
        url: str,
    ) -> Document:
        """
        Extract clean text from HTML content and return a document.

        Processes raw HTML to strip boilerplates, normalize spacing,
        and filter noise to isolate the primary textual content.

        Args:
            html: The raw HTML content to be cleaned.
            url: The origin URL of the processed HTML.

        Returns:
            A Document object containing clean text and metadata.
        """


class CacheBackend(ABC):
    """Interface for persistent cache backends storing search data."""

    @abstractmethod
    def get(
        self,
        key: str,
    ) -> list[Document] | None:
        """
        Retrieve cached documents associated with the given key.

        Args:
            key: The unique string identifier for the cached resource.

        Returns:
            A list of cached Document objects if found, otherwise None.
        """

    @abstractmethod
    def set(
        self,
        key: str,
        value: list[Document],
    ) -> None:
        """
        Store documents in the cache under a specific key.

        Args:
            key: The unique string identifier for the resource.
            value: A list of Document objects to be serialized and stored.
        """

    @abstractmethod
    def clear(self) -> None:
        """Remove all cached entries from the storage backend."""


class LLMClient(Protocol):
    """Interface for LLM keyword extraction clients."""

    def extract_keywords(
        self,
        text: str,
        limit: int,
    ) -> list[str]:
        """
        Extract a structured list of keywords from text using an LLM.

        Args:
            text: The context or text data to analyze.
            limit: The maximum number of keywords to extract.

        Returns:
            A list of extracted keyword strings.
        """

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
            text (str): The raw input text from which keywords should be extracted.

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
            query (SearchQuery): The normalized search query object.
            limit (int): The maximum number of results to return. Defaults to 10.

        Returns:
            list[SearchResult]: A list of SearchResult objects matching the query.
        """


class Downloader(ABC):
    """Interface for downloading raw web page content."""

    @abstractmethod
    def download(self, url: str) -> str:
        """
        Download raw HTML content from the specified URL.

        Args:
            url (str): The absolute HTTP/HTTPS URL of the target web page.

        Returns:
            str: The raw HTML content as a string.
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
            html (str): The raw HTML content to be cleaned.
            url (str): The origin URL of the processed HTML.

        Returns:
            Document: A Document object containing clean text and metadata.
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
            key (str): The unique string identifier for the cached resource.

        Returns:
            list[Document]: A list of cached Document objects if found, otherwise None.
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
            key (str): The unique string identifier for the resource.
            value (list[Document]): A list of Document objects to be
                serialized and stored.

        Returns:
            None
        """

    @abstractmethod
    def clear(self) -> None:
        """
        Remove all cached entries from the storage backend.

        Returns:
            None
        """


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
            text (str): The context or text data to analyze.
            limit (int): The maximum number of keywords to extract.

        Returns:
            list[str]: A list of extracted keyword strings.
        """

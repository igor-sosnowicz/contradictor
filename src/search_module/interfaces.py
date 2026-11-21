"""Module with search module interfaces."""

from abc import abstractmethod
from types import TracebackType
from typing import Self

from src.data_models.abstract.configurable import ConfigurationProvider
from src.search_module.models import Document, SearchQuery, SearchResult


class KeywordExtractor(ConfigurationProvider):
    """Interface for extracting search keywords from unstructured text."""

    @abstractmethod
    def extract(self, text: str) -> SearchQuery:
        """
        Extract keywords and return a normalised search query.

        Args:
            text (str): The raw input text from which keywords should be extracted.

        Returns:
            SearchQuery: A search query containing extracted and normalised keywords.
        """


class SearchEngine(ConfigurationProvider):
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
            query (SearchQuery): The normalised search query object.
            limit (int): The maximum number of results to return. Defaults to 10.

        Returns:
            list[SearchResult]: A list of SearchResult objects matching the query.
        """


class Downloader(ConfigurationProvider):
    """Interface for downloading raw web page content."""

    @abstractmethod
    async def __aenter__(self) -> Self:
        """
        Reserve access to resource or create a resource pool.

        Returns:
            Self: The instance of a downloader.
        """

    @abstractmethod
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> bool | None:
        """
        Release access to the resource or shut down the resource pool.

        Args:
            exc_type (type[BaseException] | None): Exception type, if an exception was
                raised in the context. Otherwise, None.
            exc (BaseException | None): Exception instance, if one was raised.
                Otherwise, None.
            _tb (TracebackType | None): Traceback, if an exception was raised.
                Otherwise, None.

        Returns:
            bool | None: Return True to suppress the exception; otherwise, return
            False or None.
        """

    @abstractmethod
    async def download(self, url: str) -> str:
        """
        Download raw HTML content from the specified URL.

        Args:
            url (str): The absolute HTTP/HTTPS URL of the target web page.

        Returns:
            str: The raw HTML content as a string.
        """


class Cleaner(ConfigurationProvider):
    """Interface for converting raw HTML into structured, readable documents."""

    @abstractmethod
    def clean(
        self,
        html: str,
        url: str,
    ) -> Document:
        """
        Extract clean text from HTML content and return a document.

        Processes raw HTML to strip boilerplates, normalise spacing,
        and filter noise to isolate the primary textual content.

        Args:
            html (str): The raw HTML content to be cleaned.
            url (str): The origin URL of the processed HTML.

        Returns:
            Document: A Document object containing clean text and metadata.
        """


class CacheBackend(ConfigurationProvider):
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
            list[Document] | None: A list of cached Document objects if found,
                otherwise None.
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
                serialised and stored.
        """

    @abstractmethod
    def clear(self) -> None:
        """Remove all cached entries from the storage backend."""

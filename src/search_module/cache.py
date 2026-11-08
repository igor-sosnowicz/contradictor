"""Disk based cache implementation."""

from types import TracebackType
from typing import Self, override

from diskcache import Cache

from src.data_models.abstract.configuration import ConfigurationValue
from src.search_module.config import CacheConfig
from src.search_module.interfaces import CacheBackend
from src.search_module.models import Document


class DiskCacheBackend(CacheBackend):
    """
    Persistent cache using diskcache.

    Stores search results on disk with expiration time.
    """

    def __init__(
        self,
        config: CacheConfig,
    ) -> None:
        """
        Initialize the disk cache backend.

        Args:
            config (CacheConfig): Configuration for the cache backend.
        """
        self.config = config
        self.cache = Cache(
            directory=config.directory,
        )

    @override
    def get(
        self,
        key: str,
    ) -> list[Document] | None:
        if not self.config.enabled:
            return None
        cached = self.cache.get(key)
        if cached is None:
            return None
        return [Document.model_validate(item) for item in cached]

    @override
    def set(
        self,
        key: str,
        value: list[Document],
    ) -> None:
        if not self.config.enabled:
            return
        serialized = [document.model_dump() for document in value]
        self.cache.set(
            key,
            serialized,
            expire=self.config.ttl_seconds,
        )

    def clear(self) -> None:
        """Remove all cached entries."""
        self.cache.clear()

    def __enter__(self) -> Self:
        """Return the cache backend as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the underlying disk cache."""
        self.cache.close()

    @override
    def get_configuration(self) -> dict[str, ConfigurationValue]:
        """Get configuration of the disk cache backend."""
        return {
            "implementation": self.__class__.__name__,
            "enabled": self.config.enabled,
            "ttl_seconds": self.config.ttl_seconds,
            "directory": self.config.directory,
        }

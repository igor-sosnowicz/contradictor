"""Tests for cache in search module."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from src.search_module.cache import DiskCacheBackend
from src.search_module.config import CacheConfig
from src.search_module.models import Document


@pytest.fixture
def cache(tmp_path: Path) -> Iterator[DiskCacheBackend]:
    """Provide a temporary disk cache and close it after the test."""
    config = CacheConfig(
        directory=str(tmp_path),
        enabled=True,
        ttl_seconds=60,
    )

    with DiskCacheBackend(config) as cache:
        yield cache


def test_cache_set_and_get(cache: DiskCacheBackend) -> None:
    """Verify that cached documents can be stored and retrieved."""
    documents = [
        Document(
            url="https://example.com",
            text="Cats are great pets.",
        )
    ]

    cache.set("cats", documents)

    cached = cache.get("cats")

    assert cached is not None
    assert len(cached) == 1
    assert cached[0].url == documents[0].url
    assert cached[0].text == documents[0].text


def test_cache_returns_none_for_missing_key(
    cache: DiskCacheBackend,
) -> None:
    """Verify that missing cache keys return None."""
    assert cache.get("missing") is None


def test_cache_clear(cache: DiskCacheBackend) -> None:
    """Verify that clearing cache removes stored entries."""
    cache.set(
        "key",
        [
            Document(
                url="https://example.com",
                text="text",
            )
        ],
    )

    cache.clear()

    assert cache.get("key") is None


def test_disabled_cache(tmp_path: Path) -> None:
    """Verify that disabled cache does not store or retrieve data."""
    config = CacheConfig(
        directory=str(tmp_path),
        enabled=False,
    )

    with DiskCacheBackend(config) as cache:
        docs = [
            Document(
                url="https://example.com",
                text="abc",
            )
        ]

        cache.set("key", docs)

        assert cache.get("key") is None

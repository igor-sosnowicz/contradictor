"""Unit tests for concurrent URL downloading in SearchPipeline."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.search_module.config import ConcurrencyConfig
from src.search_module.downloader import HttpxDownloader
from src.search_module.models import Document, SearchResult
from src.search_module.pipeline import (
    SearchPipeline,
    SearchPipelineDependencies,
)
from src.utils.errors import ContradictorError


def make_pipeline(
    urls: list[str],
    max_workers: int = 2,
) -> SearchPipeline:
    """Create a pipeline with mocked dependencies."""
    keyword_extractor = Mock()
    keyword_extractor.extract.configure_mock(
        return_value=SimpleNamespace(
            keywords=["test"],
            normalized="test query",
        )
    )

    search_engine = Mock()
    search_engine.search.configure_mock(
        return_value=[SearchResult(url=url, title="Test result") for url in urls]
    )

    downloader = Mock()
    downloader.download = AsyncMock(side_effect=lambda url: f"<html>{url}</html>")

    cleaner = Mock()
    cleaner.clean.configure_mock(
        side_effect=lambda html, url: Document(
            url=url,
            title="Test document",
            text=f"Content from {url}",
        )
    )

    cache = Mock()
    cache.get.configure_mock(return_value=None)

    return SearchPipeline(
        keyword_extractor=keyword_extractor,
        search_engine=search_engine,
        downloader=downloader,
        cleaner=cleaner,
        cache=cache,
        concurrency=ConcurrencyConfig(
            max_workers=max_workers,
        ),
    )


@pytest.mark.asyncio
async def test_downloads_run_concurrently() -> None:
    """Verify that two URLs can be downloaded concurrently."""
    barrier = asyncio.Barrier(2)

    async def fake_download(url: str) -> str:
        await asyncio.wait_for(barrier.wait(), timeout=3)
        return f"<html>{url}</html>"

    downloader = Mock()
    downloader.download = AsyncMock(side_effect=fake_download)

    pipeline = SearchPipeline(
        SearchPipelineDependencies(
            keyword_extractor=Mock(),
            search_engine=Mock(),
            downloader=downloader,
            cleaner=Mock(),
            cache=Mock(),
            concurrency=ConcurrencyConfig(),
        )
    )

    results = [
        Mock(url="https://example.com/1"),
        Mock(url="https://example.com/2"),
    ]

    downloaded = await asyncio.gather(
        *(pipeline._download_result(result) for result in results)
    )

    assert len(downloaded) == len(results)
    assert downloader.download.await_count == len(results)


@pytest.mark.asyncio
async def test_preserves_result_order() -> None:
    """Documents preserve search result order."""
    urls = [
        "https://example.com/1",
        "https://example.com/2",
        "https://example.com/3",
    ]

    pipeline = make_pipeline(urls, max_workers=3)

    barrier = asyncio.Barrier(3)

    async def fake_download(url: str) -> str:
        await barrier.wait()

        if url.endswith("/1"):
            await asyncio.sleep(0.2)
        elif url.endswith("/2"):
            await asyncio.sleep(0.1)

        return f"<html>{url}</html>"

    pipeline.downloader.download = AsyncMock(side_effect=fake_download)

    documents = await pipeline.search("test query")

    assert [document.url for document in documents] == urls


@pytest.mark.asyncio
async def test_single_url_failure_does_not_stop_pipeline() -> None:
    """One failed download does not stop other downloads."""
    urls = [
        "https://example.com/1",
        "https://example.com/2",
        "https://example.com/3",
    ]

    pipeline = make_pipeline(urls)

    async def fake_download(url: str) -> str:
        if url.endswith("/2"):
            raise ContradictorError("Download failed")

        return f"<html>{url}</html>"

    pipeline.downloader.download = AsyncMock(side_effect=fake_download)

    documents = await pipeline.search("test query")

    assert [document.url for document in documents] == [
        urls[0],
        urls[2],
    ]

    assert pipeline.downloader.download.await_count == len(urls)
    assert pipeline.cleaner.clean.call_count == len(documents)


@pytest.mark.asyncio
async def test_respects_max_workers_limit() -> None:
    """Concurrent downloads never exceed max_workers."""
    urls = [f"https://example.com/{i}" for i in range(8)]

    max_workers = 2
    pipeline = make_pipeline(urls, max_workers=max_workers)

    active_downloads = 0
    peak_downloads = 0
    lock = asyncio.Lock()
    barrier = asyncio.Barrier(max_workers)

    async def fake_download(url: str) -> str:
        nonlocal active_downloads, peak_downloads

        async with lock:
            active_downloads += 1
            peak_downloads = max(
                peak_downloads,
                active_downloads,
            )

        try:
            if url.endswith(("/0", "/1")):
                await asyncio.wait_for(
                    barrier.wait(),
                    timeout=5,
                )

            await asyncio.sleep(0.05)
            return f"<html>{url}</html>"
        finally:
            async with lock:
                active_downloads -= 1

    pipeline.downloader.download = AsyncMock(side_effect=fake_download)

    documents = await pipeline.search("test query")

    expected_documents = 8

    assert len(documents) == expected_documents
    assert peak_downloads == max_workers


@pytest.mark.asyncio
async def test_cache_hit_skips_download() -> None:
    """Cached results skip searching and downloading."""
    urls = ["https://example.com/1"]

    pipeline = make_pipeline(urls)

    cached_documents = [
        Document(
            url=urls[0],
            title="Cached document",
            text="Cached content",
        )
    ]

    pipeline.cache.get.configure_mock(return_value=cached_documents)

    documents = await pipeline.search("test query")

    assert documents == cached_documents

    pipeline.search_engine.search.assert_not_called()
    pipeline.downloader.download.assert_not_awaited()
    pipeline.cleaner.clean.assert_not_called()
    pipeline.cache.set.assert_not_called()


@pytest.mark.asyncio
async def test_httpx_downloader_closes_client() -> None:
    """Verify that the HTTP client can be closed."""
    config = Mock()
    config.configure_mock(
        user_agent="TestAgent/1.0",
        timeout=10,
    )

    downloader = HttpxDownloader(config=config)

    assert downloader._client.headers["User-Agent"] == "TestAgent/1.0"

    await downloader.close()

    assert downloader._client.is_closed

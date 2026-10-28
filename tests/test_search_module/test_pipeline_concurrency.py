"""Unit tests for concurrent URL downloading in SearchPipeline."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from time import sleep
from types import SimpleNamespace
from unittest.mock import Mock

from src.search_module.models import Document, SearchResult
from src.search_module.pipeline import SearchPipeline
from src.utils.errors import ContradictorError


def test_downloads_run_concurrently() -> None:
    """Verify that two URLs can be downloaded concurrently."""
    barrier = Barrier(2)

    def fake_download(url: str) -> str:
        barrier.wait(timeout=3)
        return f"<html>{url}</html>"

    downloader = Mock()
    downloader.download.configure_mock(side_effect=fake_download)

    pipeline = SearchPipeline(
        keyword_extractor=Mock(),
        search_engine=Mock(),
        downloader=downloader,
        cleaner=Mock(),
        cache=Mock(),
        max_workers=2,
    )

    results = [
        Mock(url="https://example.com/1"),
        Mock(url="https://example.com/2"),
    ]

    with ThreadPoolExecutor(max_workers=2) as executor:
        downloaded = list(executor.map(pipeline._download_result, results))

    expected_downloads = len(results)

    assert len(downloaded) == expected_downloads
    assert downloader.download.call_count == expected_downloads


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
    downloader.download.configure_mock(side_effect=lambda url: f"<html>{url}</html>")

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
        max_workers=max_workers,
    )


def test_preserves_result_order() -> None:
    """Documents preserve search result order."""
    urls = [
        "https://example.com/1",
        "https://example.com/2",
        "https://example.com/3",
    ]

    pipeline = make_pipeline(urls, max_workers=3)

    barrier = Barrier(3)

    def fake_download(url: str) -> str:
        barrier.wait(timeout=5)

        if url.endswith("/1"):
            sleep(0.2)
        elif url.endswith("/2"):
            sleep(0.1)

        return f"<html>{url}</html>"

    pipeline.downloader.download.configure_mock(
        side_effect=fake_download,
    )

    documents = pipeline.search("test query")

    assert [document.url for document in documents] == urls


def test_single_url_failure_does_not_stop_pipeline() -> None:
    """One failed download does not stop other downloads."""
    urls = [
        "https://example.com/1",
        "https://example.com/2",
        "https://example.com/3",
    ]

    pipeline = make_pipeline(urls)

    def fake_download(url: str) -> str:
        if url.endswith("/2"):
            raise ContradictorError("Download failed")

        return f"<html>{url}</html>"

    pipeline.downloader.download.configure_mock(
        side_effect=fake_download,
    )

    documents = pipeline.search("test query")

    assert [document.url for document in documents] == [
        urls[0],
        urls[2],
    ]

    assert pipeline.downloader.download.call_count == len(urls)
    assert pipeline.cleaner.clean.call_count == len(documents)


def test_respects_max_workers_limit() -> None:
    """Concurrent downloads never exceed max_workers."""
    urls = [f"https://example.com/{i}" for i in range(8)]

    pipeline = make_pipeline(urls, max_workers=2)

    lock = Lock()
    active_downloads = 0
    peak_downloads = 0
    barrier = Barrier(2)

    def fake_download(url: str) -> str:
        nonlocal active_downloads, peak_downloads

        with lock:
            active_downloads += 1
            peak_downloads = max(
                peak_downloads,
                active_downloads,
            )

        try:
            if url.endswith(("/0", "/1")):
                barrier.wait(timeout=5)

            sleep(0.05)
            return f"<html>{url}</html>"
        finally:
            with lock:
                active_downloads -= 1

    pipeline.downloader.download.configure_mock(side_effect=fake_download)

    documents = pipeline.search("test query")

    max_workers = 2
    expected_documents = 8

    assert len(documents) == expected_documents
    assert peak_downloads == max_workers


def test_cache_hit_skips_download() -> None:
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

    documents = pipeline.search("test query")

    assert documents == cached_documents

    pipeline.search_engine.search.assert_not_called()
    pipeline.downloader.download.assert_not_called()
    pipeline.cleaner.clean.assert_not_called()
    pipeline.cache.set.assert_not_called()

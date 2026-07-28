"""Tests for cleaner in search module."""

from pathlib import Path

import pytest

from src.search_module.cleaner import BeautifulSoupCleaner
from src.search_module.config import CleaningConfig

DATA_DIR = Path(__file__).parent / "data"


def get_html_samples() -> list[Path]:
    """Return list of all .html files from data folder as Path objects."""
    return list(DATA_DIR.glob("*.html"))


@pytest.fixture
def cleaner() -> BeautifulSoupCleaner:
    """Fixture which makes a universal cleaner object."""
    return BeautifulSoupCleaner(
        CleaningConfig(
            min_text_length=0,
            min_word_count=0,
            min_sentence_count=0,
        )
    )


def test_empty_html(cleaner: BeautifulSoupCleaner) -> None:
    """Verify that empty HTML produces an empty document."""
    document = cleaner.clean("", "https://example.com")
    assert document.text == ""


@pytest.mark.parametrize("html_path", get_html_samples(), ids=lambda p: p.name)
def test_extract_title(cleaner: BeautifulSoupCleaner, html_path: Path) -> None:
    """Verify that page title is extracted from HTML content."""
    html_content = html_path.read_text(encoding="utf-8")
    document = cleaner.clean(html_content, "https://example.com")
    assert document.title is not None
    assert len(document.title) > 0


@pytest.mark.parametrize("html_path", get_html_samples(), ids=lambda p: p.name)
def test_remove_navigation(cleaner: BeautifulSoupCleaner, html_path: Path) -> None:
    """Verify that navigation elements are removed from cleaned content."""
    html_content = html_path.read_text(encoding="utf-8")
    document = cleaner.clean(html_content, "https://example.com")
    assert "MENU" not in document.text

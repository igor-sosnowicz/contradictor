"""Tests for ReadabilityCleaner in search module."""

from pathlib import Path

import pytest

from src.search_module.cleaner import ReadabilityCleaner
from src.search_module.config import CleaningConfig

DATA_DIR = Path(__file__).parent / "html_files"


def get_html_samples() -> list[Path]:
    """Return list of all .html files from data folder as Path objects."""
    return list(DATA_DIR.glob("*.html"))


@pytest.fixture
def cleaner() -> ReadabilityCleaner:
    """Fixture which makes a universal ReadabilityCleaner with zeroed thresholds."""
    return ReadabilityCleaner(
        CleaningConfig(
            min_text_length=0,
            max_text_length=100_000,
            min_word_count=0,
            min_sentence_count=0,
            unwanted_keywords=(),
        )
    )


def test_empty_html(cleaner: ReadabilityCleaner) -> None:
    """Verify that empty HTML produces an empty document."""
    document = cleaner.clean("", "https://example.com")
    assert document.text == ""


@pytest.mark.parametrize("html_path", get_html_samples(), ids=lambda p: p.name)
def test_extract_title(cleaner: ReadabilityCleaner, html_path: Path) -> None:
    """Verify that page title is extracted from HTML content."""
    html_content = html_path.read_text(encoding="utf-8")
    document = cleaner.clean(html_content, "https://example.com")
    assert document.title is not None
    assert len(document.title) > 0


@pytest.mark.parametrize("html_path", get_html_samples(), ids=lambda p: p.name)
def test_remove_navigation(cleaner: ReadabilityCleaner, html_path: Path) -> None:
    """Verify that navigation elements are removed from cleaned content."""
    html_content = html_path.read_text(encoding="utf-8")
    document = cleaner.clean(html_content, "https://example.com")
    text_lower = document.text.lower()
    assert "menu" not in text_lower


def test_post_validation_rejects_unwanted_keywords() -> None:
    """Verify that configuration validation rejects documents with unwanted keywords."""
    config = CleaningConfig(unwanted_keywords=("login", "cookie"))
    cleaner = ReadabilityCleaner(config=config)
    html_content = (
        "<html><body><main><p>Przejdź do strony login.</p></main></body></html>"
    )
    document = cleaner.clean(html_content, "https://example.com")
    assert document.text == ""


def test_post_validation_rejects_short_text() -> None:
    """Verify that configuration validation rejects documents shorter than threshold."""
    config = CleaningConfig(min_text_length=500)
    cleaner = ReadabilityCleaner(config=config)
    html_content = "<html><body><main><p>Krótki text artykułu.</p></main></body></html>"
    document = cleaner.clean(html_content, "https://example.com")
    assert document.text == ""

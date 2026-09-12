"""Benchmark the current HTML cleaner against test fixtures."""
# ruff: noqa: T201, file won't be merged to main anyway

import json
from pathlib import Path

import pytest

from src.search_module.cleaner import BeautifulSoupCleaner
from src.search_module.config import CleaningConfig

FIXTURES_DIR = Path("tests/test_search_module/fixtures")
RESULTS_DIR = Path("benchmark-results")
RESULTS_FILE = RESULTS_DIR / "current_cleaner.json"


@pytest.mark.parametrize(
    "html_file",
    sorted(FIXTURES_DIR.glob("*.html")),
)
def test_current_cleaner(html_file: Path) -> None:
    """Run the current cleaner against an HTML fixture and save the result."""
    html = html_file.read_text(encoding="utf-8")

    cleaner = BeautifulSoupCleaner(
        config=CleaningConfig(min_text_length=200),
    )

    document = cleaner.clean(
        html=html,
        url=html_file.stem,
    )

    print(f"\n{'=' * 80}")
    print(html_file.name)
    print(f"title: {document.title}")
    print(f"characters: {len(document.text)}")
    print(f"words: {len(document.text.split())}")
    print("-" * 80)
    print(document.text[:1000])

    _save_result(
        html_file=html_file,
        title=document.title,
        text=document.text,
    )

    assert document.text


def _save_result(
    html_file: Path,
    title: str,
    text: str,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []

    if RESULTS_FILE.exists():
        results = json.loads(
            RESULTS_FILE.read_text(encoding="utf-8"),
        )

    result = {
        "file": html_file.name,
        "title": title,
        "characters": len(text),
        "words": len(text.split()),
        "text": text,
    }

    results = [existing for existing in results if existing["file"] != html_file.name]
    results.append(result)

    RESULTS_FILE.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

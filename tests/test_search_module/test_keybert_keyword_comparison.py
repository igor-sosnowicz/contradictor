"""Benchmark KeyBERT keyword extraction across different HTML cleaners."""

import json
from pathlib import Path

import pytest

from src.search_module.cleaner import BeautifulSoupCleaner
from src.search_module.cleaner_readability import ReadabilityCleaner
from src.search_module.config import CleaningConfig, KeywordConfig
from src.search_module.keyword_extraction.keybert import KeyBERTKeywordExtractor

FIXTURES_DIR = Path("tests/test_search_module/fixtures")
RESULTS_DIR = Path("benchmark-results")
RESULTS_FILE = RESULTS_DIR / "keybert_keyword_comparison.json"


@pytest.mark.parametrize(
    "html_file",
    sorted(FIXTURES_DIR.glob("*.html")),
)
def test_keybert_keyword_extraction(
    html_file: Path,
) -> None:
    """Compare KeyBERT keyword extraction across different HTML cleaners."""
    html = html_file.read_text(encoding="utf-8")

    keyword_config = KeywordConfig(
        max_keywords=5,
    )

    extractor = KeyBERTKeywordExtractor(
        config=keyword_config,
    )

    cleaners = {
        "current": BeautifulSoupCleaner(
            config=CleaningConfig(min_text_length=200),
        ),
        "readability": ReadabilityCleaner(),
    }

    result: dict[str, object] = {
        "file": html_file.name,
        "cleaners": {},
    }

    for cleaner_name, cleaner in cleaners.items():
        document = cleaner.clean(
            html=html,
            url=html_file.stem,
        )

        query = extractor.extract(document.text)

        result["cleaners"][cleaner_name] = {
            "title": document.title,
            "characters": len(document.text),
            "words": len(document.text.split()),
            "keywords": list(query.keywords),
            "normalized": query.normalized,
        }

    _save_result(result)


def _save_result(result: dict[str, object]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []

    if RESULTS_FILE.exists():
        results = json.loads(
            RESULTS_FILE.read_text(encoding="utf-8"),
        )

    file_name = result["file"]

    results = [existing for existing in results if existing["file"] != file_name]

    results.append(result)

    results.sort(
        key=lambda item: str(item["file"]),
    )

    RESULTS_FILE.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

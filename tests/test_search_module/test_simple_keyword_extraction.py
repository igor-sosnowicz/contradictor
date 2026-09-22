"""Benchmark simple keyword extraction using ReadabilityCleaner."""

import json
import re
from pathlib import Path

import pytest

from src.search_module.cleaner import ReadabilityCleaner
from src.search_module.config import CleaningConfig, KeywordConfig
from src.search_module.keyword_extraction.simple import SimpleKeywordExtractor
from src.utils.tokenisers.tokeniser import Tokeniser

BASE_DIR = Path(__file__).resolve().parents[2]
FIXTURES_DIR = BASE_DIR / "tests" / "test_search_module" / "fixtures"
RESULTS_DIR = BASE_DIR / "benchmark-results-readability"
RESULTS_FILE = RESULTS_DIR / "simple_readability_keywords.json"


class BenchmarkTokeniser(Tokeniser):
    """Tokeniser used for benchmarking the simple keyword extractor."""

    def tokenise(self, text: str) -> list[str]:
        """Split text into lowercase words and remove common stop words."""
        words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
        return [
            word
            for word in words
            if word
            not in {
                "a",
                "an",
                "and",
                "are",
                "as",
                "at",
                "be",
                "because",
                "by",
                "for",
                "from",
                "in",
                "is",
                "it",
                "of",
                "on",
                "or",
                "that",
                "the",
                "this",
                "to",
                "was",
                "were",
                "with",
            }
        ]

    def encode(self, text: str) -> list[str]:
        """Encode text into tokens using the benchmark tokenisation."""
        return self.tokenise(text)

    def decode(self, tokens: list[str]) -> str:
        """Decode tokens into a space-separated string."""
        return " ".join(tokens)


@pytest.mark.parametrize(
    "html_file",
    sorted(FIXTURES_DIR.glob("*.html")) if FIXTURES_DIR.exists() else [],
)
def test_simple_keyword_extraction(
    html_file: Path,
) -> None:
    """Evaluate SimpleKeywordExtractor on text extracted by ReadabilityCleaner."""
    html = html_file.read_text(encoding="utf-8")
    keyword_config = KeywordConfig(
        max_keywords=5,
        stop_words={
            "are",
            "the",
            "and",
            "because",
            "that",
            "this",
        },
    )
    tokeniser = BenchmarkTokeniser()
    extractor = SimpleKeywordExtractor(
        config=keyword_config,
        tokeniser=tokeniser,
    )
    cleaning_config = CleaningConfig(min_text_length=150)
    cleaner = ReadabilityCleaner(config=cleaning_config)
    document = cleaner.clean(
        html=html,
        url=html_file.stem,
    )
    query = extractor.extract(document.text)
    result: dict[str, object] = {
        "file": html_file.name,
        "title": document.title,
        "metrics": {
            "characters": len(document.text),
            "words": len(document.text.split()),
        },
        "extraction": {
            "keywords": list(query.keywords),
            "normalized": query.normalized,
        },
    }
    _save_result(result)


def _save_result(result: dict[str, object]) -> None:
    """Persist extraction results incrementally into a JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    if RESULTS_FILE.exists():
        try:
            results = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            results = []
    file_name = result["file"]
    results = [existing for existing in results if existing["file"] != file_name]
    results.append(result)
    results.sort(key=lambda item: str(item["file"]))
    RESULTS_FILE.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

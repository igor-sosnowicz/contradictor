"""Benchmark n-gram keyword extraction using ReadabilityCleaner."""

import json
import re
from pathlib import Path

import pytest

from src.search_module.cleaner import ReadabilityCleaner
from src.search_module.config import CleaningConfig, KeywordConfig
from src.search_module.keyword_extraction.ngram import NGramKeywordExtractor
from src.utils.tokenisers.tokeniser import Tokeniser

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FIXTURES_DIR = BASE_DIR / "tests" / "test_search_module" / "fixtures"
RESULTS_DIR = BASE_DIR / "benchmark-results-readability"
RESULTS_FILE = RESULTS_DIR / "ngram_readability_keywords.json"


class BenchmarkTokeniser(Tokeniser):
    """Tokeniser used for benchmarking the n-gram keyword extractor."""

    def encode(self, text: list[str]) -> list[list[int]]:
        """Encode each token into a list of character code points."""
        return [[ord(char) for char in item] for item in text]

    def decode(self, tokens: list[list[int]]) -> list[str]:
        """Decode character code points back into tokens."""
        return ["".join(chr(token) for token in item) for item in tokens]

    def tokenise(self, text: str) -> list[str]:
        """Extract lowercase alphabetic words from the input text."""
        return re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())


@pytest.mark.parametrize(
    "html_file",
    sorted(FIXTURES_DIR.glob("*.html")) if FIXTURES_DIR.exists() else [],
)
def test_ngram_keyword_extraction(
    html_file: Path,
) -> None:
    """Evaluate NGramKeywordExtractor on text extracted by ReadabilityCleaner."""
    html = html_file.read_text(encoding="utf-8")
    keyword_config = KeywordConfig(
        max_keywords=5,
        min_ngram_size=2,
        max_ngram_size=4,
        stop_words={
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
        },
    )
    tokeniser = BenchmarkTokeniser()
    extractor = NGramKeywordExtractor(
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

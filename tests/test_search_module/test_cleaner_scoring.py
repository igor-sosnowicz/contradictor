"""Tests for determining noise_penalty_factor in search module."""

import sys
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from src.search_module.cleaner import BeautifulSoupCleaner
from src.search_module.config import CleaningConfig

DATA_DIR = Path(__file__).parent / "html_files"


@pytest.fixture
def cleaner() -> BeautifulSoupCleaner:
    """
    Fixture initializing the cleaner for snapshots.

    Uses zeroed text length thresholds to prevent truncation errors.
    """
    config = CleaningConfig(
        noise_penalty_factor=0.15,
        min_text_length=0,
        min_word_count=0,
        min_sentence_count=0,
    )
    config.min_text_length = 0
    config.min_word_count = 0
    config.min_sentence_count = 0
    return BeautifulSoupCleaner(config=config)


def _load_html_file(filename: str) -> str:
    """Read HTML content from the data directory."""
    file_path = DATA_DIR / filename
    return file_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "html_filename",
    [
        "wikipedia_python.html",
        "medium_engineering.html",
        "python_org_about.html",
    ],
)
def test_cleaner_scoring_with_real_html_fixtures(
    cleaner: BeautifulSoupCleaner,
    html_filename: str,
) -> None:
    """
    Verify cleaner logic and scoring system.

    Uses real-world website HTML fixtures for regression testing.
    """
    html_content = _load_html_file(html_filename)

    document = cleaner.clean(html_content, url=html_filename)
    assert document.text != ""
    assert len(document.text) >= cleaner.config.min_text_length

    soup = BeautifulSoup(html_content, "html.parser")
    candidates = cleaner._extract_candidates(soup)

    assert len(candidates) > 0, f"No content candidates found in {html_filename}"

    best_candidate = cleaner._select_best_candidate(candidates)
    assert cleaner._content_score(best_candidate) > 0.0


def test_grid_search_noise_penalty_factor_pure() -> None:
    """Mathematical verification of the noise penalty factor with file logging."""
    clean_article = (
        "Python is an interpreted high-level programming language. "
        "Its design philosophy emphasizes code readability "
        "with use of significant indentation. "
        "Its language constructs as well as its object-oriented approach "
        "aim to help programmers write clear, logical code for small "
        "and large-scale projects."
    )

    noise_block = (
        "Sign up for our newsletter. Cookie privacy policy. "
        "Related posts and comments login. Share on social media. Subscribe now."
    )
    output_file = Path(__file__).parent / "results.txt"
    lines_to_save = []
    header_footer = "==============================================\n"
    title_line = "=== NOISE PENALTY FACTOR OPTIMIZATION ANALYSIS ===\n"

    lines_to_save.append(title_line)
    for factor in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
        config = CleaningConfig(
            noise_penalty_factor=factor,
            min_text_length=0,
            min_word_count=0,
            min_sentence_count=0,
        )
        cleaner = BeautifulSoupCleaner(config=config)
        article_score = cleaner._content_score(clean_article)
        noise_score = cleaner._content_score(noise_block)
        margin = article_score - noise_score
        is_winner_correct = article_score > noise_score
        result_line = (
            f"Factor: {factor:.2f} -> Article Score: {article_score:>5.1f} | "
            f"Noise Score: {noise_score:>5.1f} | Selection Margin: {margin:>6.1f} | "
            f"Correct Winner? {is_winner_correct!s}\n"
        )
        lines_to_save.append(result_line)
    lines_to_save.append(header_footer)

    output_file.write_text("".join(lines_to_save), encoding="utf-8")
    sys.stderr.write(f"\nSaved results to: {output_file.resolve()}\n")
    sys.stderr.write(header_footer)
    for line in lines_to_save:
        sys.stderr.write(line)

    assert True

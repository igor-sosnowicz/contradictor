"""Tests for sentence splitter."""

from src.utils.sentence_splitter import (
    SentenceSplitter,
)

EXPECTED_SENTENCE_COUNT = 2


def test_sentence_splitter_splits_text() -> None:
    """Verify sentences are separated."""
    splitter = SentenceSplitter()

    result = list(splitter("Cats are animals. Dogs are animals."))

    assert len(result) == EXPECTED_SENTENCE_COUNT


def test_sentence_splitter_handles_empty_text() -> None:
    """Verify empty input."""
    splitter = SentenceSplitter()

    result = list(splitter(""))

    assert result == []

"""Tests for sentence splitter."""

from src.utils.sentence_splitter import (
    SentenceSplitter,
)


def test_sentence_splitter_splits_text() -> None:
    """Verify sentences are separated."""
    splitter = SentenceSplitter()
    expected_sentence_count = 2

    result = list(splitter("Cats are animals. Dogs are animals."))

    assert len(result) == expected_sentence_count


def test_sentence_splitter_handles_empty_text() -> None:
    """Verify empty input."""
    splitter = SentenceSplitter()

    result = list(splitter(""))

    assert result == []

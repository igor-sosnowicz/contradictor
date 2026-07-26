"""Utility functions for search module."""

import hashlib
import re


def create_cache_key(
    text: str,
) -> str:
    """Create stable cache key from input text."""
    normalized = text.lower().strip()

    normalized = " ".join(normalized.split())

    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def tokenize(
    text: str,
    min_word_length: int,
    stop_words: set[str],
) -> list[str]:
    """Tokenize text and remove stop words."""
    words = re.findall(
        rf"\b[a-zA-Z]{{{min_word_length},}}\b",
        text.lower(),
    )

    return [word for word in words if word not in stop_words]

"""Utility functions for search module."""

import hashlib


def create_cache_key(
    text: str,
) -> str:
    """Create stable cache key from input text."""
    normalized = text.lower().strip()

    normalized = " ".join(normalized.split())

    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

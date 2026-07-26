"""Utilities for cache key generation."""

import hashlib

CACHE_VERSION = "v1"


def create_cache_key(
    query: str,
) -> str:
    """Create stable cache key for search query."""
    normalized = query.lower().strip()

    payload = f"{CACHE_VERSION}:{normalized}"

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

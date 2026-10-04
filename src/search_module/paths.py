"""Paths owned by the search module."""

from enum import Enum

from src.paths.core import PathRoot, PathSpec


class SearchPaths(Enum):
    """Wishlist of the search module."""

    SEARCH_CACHE_DIR = PathSpec(PathRoot.CACHE, "search", is_dir=True)

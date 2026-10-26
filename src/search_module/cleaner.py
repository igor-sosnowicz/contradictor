"""Backwards-compatible shim for the moved cleaner module."""

from src.search_module.cleaners import NullCleaner, ReadabilityCleaner

__all__ = ["NullCleaner", "ReadabilityCleaner"]

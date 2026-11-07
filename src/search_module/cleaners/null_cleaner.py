"""Null cleaner that passes already-plain-text content through unchanged."""

from typing import override

from src.search_module.interfaces import Cleaner
from src.search_module.models import Document


class NullCleaner(Cleaner):
    """Pass-through cleaner for content that is already plain text."""

    @override
    def clean(
        self,
        html: str,
        url: str,
    ) -> Document:
        return Document(url=url, text=(html or "").strip(), title="")

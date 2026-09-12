"""Readability-based HTML cleaner."""

from typing import override

from bs4 import BeautifulSoup
from readability import parse

from src.search_module.interfaces import Cleaner
from src.search_module.models import Document


class ReadabilityCleaner(Cleaner):
    """Extract main readable content using Mozilla Readability."""

    @override
    def clean(
        self,
        html: str,
        url: str,
    ) -> Document:
        if not html.strip():
            return Document(
                url=url,
                text="",
            )

        article = parse(
            html,
            url=url,
        )

        content = article.content or ""
        text = self._html_to_text(content)

        return Document(
            url=url,
            text=text,
            title=article.title or "",
        )

    def _html_to_text(
        self,
        content: str,
    ) -> str:
        """Convert Readability HTML content to plain text."""
        if not content:
            return ""

        soup = BeautifulSoup(
            content,
            "html.parser",
        )

        return soup.get_text(
            separator=" ",
            strip=True,
        )

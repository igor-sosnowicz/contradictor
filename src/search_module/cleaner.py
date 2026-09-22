"""
HTML cleaner module that extracts primary readable text
using Mozilla Readability andvalidates it against quality thresholds.
"""

import logging
from typing import override

from bs4 import BeautifulSoup
from readability import parse

from src.search_module.config import CleaningConfig
from src.search_module.interfaces import Cleaner
from src.search_module.models import Document

logger = logging.getLogger(__name__)


class ReadabilityCleaner(Cleaner):
    """Extract main readable content using Mozilla Readability with post-validation."""

    def __init__(self, config: CleaningConfig) -> None:
        """Initialize cleaner with quality thresholds."""
        self.config = config

    @override
    def clean(
        self,
        html: str,
        url: str,
    ) -> Document:
        if not html or not html.strip():
            return Document(url=url, text="")
        try:
            article = parse(html, url=url)
            content = article.content or ""
            title = article.title or ""
            text = self._html_to_text(content)
        except (ValueError, AttributeError, TypeError) as e:
            logger.debug("Readability parsing failed for %s: %s", url, e)
            text = self._html_to_text(html)
            title = ""

        text_lower = text.lower()
        if any(keyword in text_lower for keyword in self.config.unwanted_keywords):
            logger.warning(
                "Document %s rejected: text too short (%s chars).",
                url,
                len(text),
            )
            return Document(url=url, text="", title=title)

        if len(text) < self.config.min_text_length:
            logger.warning(
                "Document %s rejected: text too short (%s chars).",
                url,
                len(text),
            )
            return Document(url=url, text="", title=title)

        if len(text) > self.config.max_text_length:
            logger.debug(
                "Document %s text truncated to %s chars.",
                url,
                self.config.max_text_length,
            )
            text = text[: self.config.max_text_length]

        word_count = len(text.split())
        if word_count < self.config.min_word_count:
            logger.warning(
                "Document %s rejected: word count too low (%s words).",
                url,
                word_count,
            )
            return Document(url=url, text="", title=title)

        return Document(
            url=url,
            text=text,
            title=title,
        )

    def _html_to_text(
        self,
        content: str,
    ) -> str:
        """
        Convert HTML content to plain text.

        Uses lxml parser and filter out structural noise.
        """
        soup = BeautifulSoup(
            content,
            "lxml",
        )
        for technical_tag in soup(["script", "style", "noscript", "form", "svg"]):
            technical_tag.decompose()
        raw_text = soup.get_text(
            separator="\n",
            strip=True,
        )
        cleaned_lines: list[str] = []
        noise_keywords = [kw.lower() for kw in self.config.noise_line_keywords]

        for line in raw_text.splitlines():
            line_stripped = line.strip()
            if not line_stripped:
                continue
            if line_stripped.startswith(("{", "}", "[", "]", "function", "var ")):
                continue
            if (
                line_stripped.startswith(("http://", "https://"))
                and " " not in line_stripped
            ):
                continue
            line_lower = line_stripped.lower()
            if any(keyword in line_lower for keyword in noise_keywords):
                continue
            cleaned_lines.append(line_stripped)
        return " ".join(cleaned_lines)

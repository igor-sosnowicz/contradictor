"""HTML cleaner implementation."""

import logging
import re

from bs4 import BeautifulSoup, Tag

from src.search_module.config import CleaningConfig
from src.search_module.interfaces import Cleaner
from src.search_module.models import Document
from src.utils.errors import ContradictorError

logger = logging.getLogger(__name__)
MIN_CANDIDATE_LENGTH = 200


class BeautifulSoupCleaner(Cleaner):
    """
    Extracts main readable content from HTML.

    Strategy:
    - remove obvious non-content elements,
    - find candidate content blocks,
    - select the most informative block,
    - normalize extracted text.
    """

    def __init__(
        self,
        config: CleaningConfig,
    ) -> None:
        """Initialize HTML cleaner with provided configuration."""
        self.config = config

    def clean(
        self,
        html: str,
        url: str,
    ) -> Document:
        """Clean HTML content and return extracted document text."""
        if not html.strip():
            return Document(
                url=url,
                text="",
            )
        soup = BeautifulSoup(
            html,
            "html.parser",
        )
        title = self._extract_title(soup)
        self._remove_unwanted_tags(soup)
        candidates = self._extract_candidates(soup)
        text = self._select_best_candidate(candidates)
        text = self._normalize_text(text)

        if not self._is_valid_content(text):
            logger.warning(
                "Rejected low quality document: %s",
                url,
            )
            text = ""

        if len(text) < self.config.min_text_length:
            logger.warning(
                "Text too short after cleaning: %s",
                url,
            )

        return Document(
            url=url,
            text=text,
            title=title,
        )

    def _extract_title(
        self,
        soup: BeautifulSoup,
    ) -> str:
        if soup.title:
            return soup.title.get_text(strip=True)
        return ""

    def _remove_unwanted_tags(
        self,
        soup: BeautifulSoup,
    ) -> None:
        for tag_name in self.config.remove_tags:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        for element in soup.find_all(["div", "section", "aside"]):
            if not isinstance(element, Tag):
                continue

            attrs = element.attrs or {}
            element_id = attrs.get(
                "id",
                "",
            )
            classes = element.get("class")

            if classes is None:
                classes = ""
            elif isinstance(classes, list):
                classes = " ".join(classes)
            else:
                classes = str(classes)

            identifier = (f"{element_id} {classes}").lower()
            if self._is_noise_element(identifier):
                element.decompose()

    def _is_noise_element(
        self,
        identifier: str,
    ) -> bool:
        noise_words = (
            "menu",
            "nav",
            "footer",
            "header",
            "cookie",
            "popup",
            "modal",
            "newsletter",
            "subscribe",
            "advert",
            "social",
            "share",
            "comment",
            "related",
            "recommend",
            "login",
            "author",
            "byline",
            "credit",
            "caption",
            "metadata",
            "published",
            "updated",
        )
        return any(word in identifier for word in noise_words)

    def _extract_candidates(
        self,
        soup: BeautifulSoup,
    ) -> list[str]:
        candidates: list[str] = []
        preferred = soup.find_all(
            [
                "article",
                "main",
            ]
        )
        for element in preferred:
            text = self._element_text(element)
            if len(text) >= MIN_CANDIDATE_LENGTH:
                candidates.append(text)
        if not candidates:
            for element in soup.find_all("div"):
                text = self._element_text(element)
                if len(text) >= MIN_CANDIDATE_LENGTH:
                    candidates.append(text)
        return candidates

    def _element_text(
        self,
        element: Tag,
    ) -> str:
        if not isinstance(
            element,
            Tag,
        ):
            return ""

        try:
            return element.get_text(
                separator=" ",
                strip=True,
            )

        except ContradictorError:
            logger.debug(
                "Failed extracting text from element",
                exc_info=True,
            )

            return ""

    def _select_best_candidate(
        self,
        candidates: list[str],
    ) -> str:
        if not candidates:
            return ""

        return max(
            candidates,
            key=self._content_score,
        )

    def _content_score(
        self,
        text: str,
    ) -> float:
        score = len(text)
        lowered = text.lower()
        noise_penalty = (
            "subscribe",
            "newsletter",
            "cookie",
            "privacy",
            "related",
            "share",
            "login",
        )
        for word in noise_penalty:
            if word in lowered:
                score -= 500
        score += text.count(".") * 10

        return score

    def _normalize_text(
        self,
        text: str,
    ) -> str:
        if self.config.collapse_whitespace:
            text = re.sub(
                r"[ \t]+",
                " ",
                text,
            )

        if self.config.remove_empty_lines:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = "\n".join(lines)

        return text.strip()

    def _is_valid_content(
        self,
        text: str,
    ) -> bool:

        if len(text) < self.config.min_text_length:
            return False
        words = text.split()

        if len(words) < self.config.min_word_count:
            return False

        sentences = text.count(".")

        return sentences >= self.config.min_sentence_count

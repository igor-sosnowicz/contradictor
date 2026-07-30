"""HTML cleaner implementation."""

import re
from typing import ClassVar, override

from bs4 import BeautifulSoup, Tag
from loguru import logger

from src.search_module.config import CleaningConfig
from src.search_module.interfaces import Cleaner
from src.search_module.models import Document
from src.utils.errors import ContradictorError


class BeautifulSoupCleaner(Cleaner):
    """
    Extracts main readable content from HTML.

    Strategy:
    - remove obvious non-content elements,
    - find candidate content blocks,
    - select the most informative block,
    - normalize extracted text.
    """

    MIN_CANDIDATE_LENGTH: ClassVar[int] = 200
    SENTENCE_SCORE_WEIGHT: ClassVar[int] = 10

    def __init__(
        self,
        config: CleaningConfig,
    ) -> None:
        """
        Initialize the HTML cleaner.

        Args:
            config (CleaningConfig): Configuration for HTML cleaning.

        Returns:
            None
        """
        self.config = config

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
                "Rejected low quality document: {}",
                url,
            )
            text = ""

        if len(text) < self.config.min_text_length:
            logger.warning(
                "Text too short after cleaning: {}",
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
        """
        Extract title from parsed HTML.

        Args:
            soup (BeautifulSoup): Parsed HTML tree.

        Returns:
            str: Page title or empty string.
        """
        if soup.title:
            return soup.title.get_text(strip=True)
        return ""

    def _remove_unwanted_tags(
        self,
        soup: BeautifulSoup,
    ) -> None:
        """
        Remove unwanted HTML elements from document.

        Args:
            soup (BeautifulSoup): Parsed HTML document to modify.

        Returns:
            None: This method modifies the document in place.
        """
        for tag_name in self.config.remove_tags:
            for tag in list(soup.find_all(tag_name)):
                tag.decompose()

        to_decompose = [
            element
            for element in soup.find_all(["div", "section", "aside"])
            if self._should_decompose(element)
        ]

        for element in to_decompose:
            try:
                element.decompose()
            except AttributeError:
                continue

    def _should_decompose(self, element: Tag) -> bool:
        """
        Check whether an HTML element should be removed.

        Args:
            element (Tag): HTML element to evaluate.

        Returns:
            bool: True if element should be removed, otherwise False.
        """
        if not isinstance(element, Tag) or element.attrs is None:
            return False

        attrs = element.attrs
        element_id = attrs.get("id", "")
        classes = attrs.get("class")

        if classes is None:
            classes = ""
        elif isinstance(classes, list):
            classes = " ".join(classes)
        else:
            classes = str(classes)

        identifier = f"{element_id} {classes}".lower()
        if not self._is_noise_element(identifier):
            return False

        element_text = self._element_text(element)
        return len(element_text) < self.MIN_CANDIDATE_LENGTH

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
            if len(text) >= self.MIN_CANDIDATE_LENGTH:
                candidates.append(text)
        if not candidates:
            for element in soup.find_all("div"):
                text = self._element_text(element)
                if len(text) >= self.MIN_CANDIDATE_LENGTH:
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
        """
        Calculate content quality score.

        Args:
            text (str): Extracted text candidate.

        Returns:
            float: Score representing estimated content quality.
        """
        base_score = len(text) + (text.count(".") * self.SENTENCE_SCORE_WEIGHT)
        lowered = text.lower()
        total_penalty_ratio = 0.0
        for word in self.config.noise_penalty_words:
            if word in lowered:
                total_penalty_ratio += self.config.noise_penalty_factor
        total_penalty_ratio = min(total_penalty_ratio, 1.0)
        return base_score * (1.0 - total_penalty_ratio)

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

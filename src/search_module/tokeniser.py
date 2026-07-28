"""Tokeniser implementations for the search module."""

import re

from src.data_models.abstract.tokeniser import Tokeniser


class SimpleTokeniser(Tokeniser):
    """Basic tokeniser implementation using regex and stop words filtering."""

    def __init__(
        self,
        min_word_length: int = 3,
        stop_words: set[str] | None = None,
    ) -> None:
        """Initialize the tokeniser with filtering rules."""
        self.min_word_length = min_word_length
        self.stop_words = stop_words or set()

    def tokenise(
        self,
        text: str,
    ) -> list[str]:
        """Tokenise text and remove stop words."""
        words = re.findall(
            rf"\b[a-zA-Z]{{{self.min_word_length},}}\b",
            text.lower(),
        )

        return [word for word in words if word not in self.stop_words]

    def encode(
        self,
        text: list[str],
    ) -> list[list[int]]:
        """Integer encoding is not supported by SimpleTokeniser."""
        raise NotImplementedError("SimpleTokeniser does not support integer encoding.")

    def decode(
        self,
        tokens: list[list[int]],
    ) -> list[str]:
        """Integer decoding is not supported by SimpleTokeniser."""
        raise NotImplementedError("SimpleTokeniser does not support integer decoding.")

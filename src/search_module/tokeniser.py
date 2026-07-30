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
        """
        Initialize tokeniser filtering configuration.

        Args:
            min_word_length (int): Minimum number of characters required for
                a word to be included in output tokens.
            stop_words (set[str] | None): Words to exclude from generated
                tokens. Defaults to None.

        Returns:
            None: Initializes tokeniser configuration.
        """
        self.min_word_length = min_word_length
        self.stop_words = stop_words or set()

    def tokenise(
        self,
        text: str,
    ) -> list[str]:
        """
        Convert text into filtered tokens.

        Extracts alphabetic words, converts them to lowercase, removes words
        shorter than the configured minimum length, and filters stop words.

        Args:
            text (str): Input text to tokenise.

        Returns:
            list[str]: List of extracted tokens.
        """
        words = re.findall(
            rf"\b[a-zA-Z]{{{self.min_word_length},}}\b",
            text.lower(),
        )

        return [word for word in words if word not in self.stop_words]

    def encode(
        self,
        text: list[str],
    ) -> list[list[int]]:
        """
        Encode tokens into integer representations.

        This implementation does not provide integer encoding.

        Args:
            text (list[str]): Tokens to encode.

        Returns:
            list[list[int]]: Encoded token representations.

        Raises:
            NotImplementedError: Always raised because integer encoding
                is not supported.
        """
        raise NotImplementedError("SimpleTokeniser does not support integer encoding.")

    def decode(
        self,
        tokens: list[list[int]],
    ) -> list[str]:
        """
        Decode integer token representations into text tokens.

        This implementation does not provide integer decoding.

        Args:
            tokens (list[list[int]]): Encoded tokens to decode.

        Returns:
            list[str]: Decoded text tokens.

        Raises:
            NotImplementedError: Always raised because integer decoding
                is not supported.
        """
        raise NotImplementedError("SimpleTokeniser does not support integer decoding.")

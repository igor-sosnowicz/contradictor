"""Module with a text tokeniser interface."""

from abc import ABC, abstractmethod


class Tokeniser(ABC):
    """Abstract text tokeniser."""

    @abstractmethod
    def encode(self, text: list[str]) -> list[list[int]]:
        """
        Encode a batch of texts into token IDs.

        Args:
            text (list[str]): List of texts to encode.

        Returns:
            list[list[int]]: A list of token ID sequences, one per input text.
        """

    @abstractmethod
    def decode(self, tokens: list[list[int]]) -> list[str]:
        """
        Decode batches of token IDs back into text.

        Args:
            tokens (list[list[int]]): List of token ID sequences.

        Returns:
            list[str]: The decoded texts, one per input token sequence.
        """

    @abstractmethod
    def tokenise(self, text: str) -> list[str]:
        """
        Split text into individual string tokens.

        Used for text cleaning, keyword extraction and n-grams.
        """
        raise NotImplementedError

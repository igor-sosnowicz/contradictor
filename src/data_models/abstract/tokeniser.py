"""Module with a text tokeniser interface."""

from abc import ABC, abstractmethod


class Tokeniser(ABC):
    """Abstract text tokeniser."""

    @abstractmethod
    def encode(self, text: list[str]) -> list[list[int]]:
        """
        Encode a batch of texts into token IDs.

        Args:
            text: List of texts to encode.

        Returns:
            A list of token ID sequences, one per input text.
        """

    @abstractmethod
    def decode(self, tokens: list[list[int]]) -> list[str]:
        """
        Decode batches of token IDs back into text.

        Args:
            tokens: List of token ID sequences.

        Returns:
            The decoded texts, one per input token sequence.
        """

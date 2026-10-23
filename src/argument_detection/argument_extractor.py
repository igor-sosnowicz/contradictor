"""Module with an interface for an argument extractor."""

from abc import ABC, abstractmethod

from src.data_models.data_models import Argument


class ArgumentExtractor(ABC):
    """An interface for an argument extractor."""

    @abstractmethod
    def __call__(self, texts: list[str]) -> list[list[Argument]]:
        """
        Extract arguments (claim+evidence) from texts.

        Args:
            texts (list[str]): Texts, from which arguments should be extracted.

        Returns:
            list[list[Argument]]: Lists of extracted arguments.
                One per input document. Order is preserved.
        """

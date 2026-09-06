"""Module with a natural language inference interface."""

from abc import ABC, abstractmethod

from src.data_models.data_models import NLIResult


class NLI(ABC):
    """An interface for a natural language inference (NLI)."""

    @abstractmethod
    async def __call__(
        self, reference_texts: list[str], other_texts: list[str]
    ) -> list[list[NLIResult]]:
        """
        Perform natural language inference for multiple reference texts at once.

        Args:
            reference_texts (list[str]): Reference texts with regard to which
                inference is performed.
            other_texts (list[str]): List of texts compared against every
                reference text.

        Returns:
            list[list[NLIResult]]: Inference results for every reference text.
                The outer list is aligned with `reference_texts`; the inner
                list is aligned with `other_texts`.
        """

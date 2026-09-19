"""Module with a natural language inference interface."""

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Self

from src.data_models.data_models import NLIPrediction


class NLI(ABC):
    """An interface for a natural language inference (NLI)."""

    @abstractmethod
    async def __call__(
        self, reference_texts: list[str], other_texts: list[str]
    ) -> list[list[NLIPrediction]]:
        """
        Perform natural language inference for multiple reference texts at once.

        Args:
            reference_texts (list[str]): Reference texts with regard to which
                inference is performed.
            other_texts (list[str]): List of texts compared against every
                reference text.

        Returns:
            list[list[NLIPrediction]]: Inference results for every reference
                text. The outer list is aligned with `reference_texts`; the inner
                list is aligned with `other_texts`. Each result contains confidence
                score in the range [0, 1].

        Raises:
            NLIResultError: When a model returned a non-existent label as a NLI result.
        """

    @abstractmethod
    def __enter__(self) -> Self:
        """
        Prepare the NLI model.

        Returns:
            Self: The prepared instance of the NLI model.
        """

    @abstractmethod
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """
        Free up resources when the NLI model is no longer needed.

        Args:
            exc_type (type[BaseException] | None): The type of an exception.
                None if exited normally.
            exc_value (BaseException | None): A value of an exception.
                None if exited normally.
            traceback (TracebackType | None): The traceback of an exception.
                None if exited normally.
        """

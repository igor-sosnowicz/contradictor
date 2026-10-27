"""Module with an interface for a unified argument and style encoder."""

from abc import ABC, abstractmethod

import numpy as np

from src.data_models.data_models import StyledFramedArgument


class ArgumentEncoder(ABC):
    """An interface for a unified argument and style encoder."""

    @abstractmethod
    async def __call__(self, arguments: list[StyledFramedArgument]) -> list[np.ndarray]:
        """
        Encode arguments and their style in a vector representation.

        The vector representation allows for fast comparisons to find
        counterarguments with a vector search.

        Args:
            arguments (list[StyledFramedArgument]): List of styled and framed arguments
                sharing the same frame.

        Returns:
            list[np.ndarray]: List of vector representations of the input arguments.
        """

    @abstractmethod
    async def prepare(self) -> None:
        """Prepare the dataset to be used."""

    @abstractmethod
    def reset(self) -> None:
        """Remove trained weights of the encoder from disk and memory."""

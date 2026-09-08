"""Module with an interface for a unified argument and style encoder."""

from abc import ABC, abstractmethod

import numpy as np

from src.data_models.data_models import FramedArgument


class Encoder(ABC):
    """An interface for a unified argument and style encoder."""

    @abstractmethod
    async def __call__(
        self, arguments: list[FramedArgument], arguments_style: list[np.ndarray]
    ) -> list[np.ndarray]:
        """
        Encode arguments and their style in a vector representation.

        The vector representation allows for fast comparisons to find
        counterarguments with a vector search.

        Args:
            arguments (list[FramedArgument]): List of framed arguments sharing
                the same frame.
            arguments_style (list[np.ndarray]): List of style vectors of the arguments.

        Returns:
            list[np.ndarray]: List of vector representations of the input arguments.
        """

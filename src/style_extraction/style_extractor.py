"""Module for enapsulating a style in a vectorised representation."""

from abc import ABC, abstractmethod

import numpy as np


class StyleExtractor(ABC):
    """Text style extractor into a deterministic vector representation."""

    @abstractmethod
    async def prepare(
        self,
    ) -> None:
        """Prepare the extractor by downloading non-module dependencies."""

    @abstractmethod
    async def extract(self, texts: list[str]) -> list[np.ndarray]:
        """
        Efficiently extract features of style in a fixed-size vector.

        Args:
            texts (list[str]): List of texts for which style should be vectorised.

        Returns:
            list[np.ndarray]: List of vectors in order matching the input texts.
        """

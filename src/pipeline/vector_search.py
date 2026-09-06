"""Module with an interface of a vector search."""

from abc import ABC, abstractmethod

import numpy as np


class VectorSearch(ABC):
    """An interface for a vector search engine."""

    @abstractmethod
    async def max_distance(
        self,
        references: list[np.ndarray],
        other: list[np.ndarray],
        max_candidates: int | None = None,
    ) -> list[list[int]]:
        """
        Get indices of `other` vectors the farthest from every reference vector.

        Args:
            references (list[np.ndarray]): Reference vectors against which all
                distances are calculated.
            other (list[np.ndarray]): List of vectors to which distance from
                the reference vectors are calculated.
            max_candidates (int | None): A maximum number of candidates to return.
                If None, defaults to the value from the configuration.

        Returns:
            list[list[int]]: For every reference vector, a list of indices of
                vectors being the farthest from it. Each list is ordered by the
                distance to the corresponding reference vector. The farthest
                ones are first. The outer list is aligned with `references`.
        """

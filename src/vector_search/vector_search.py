"""Module with an interface of a vector search."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class VectorSearch(ABC):
    """An interface for a vector search engine."""

    HARD_LIMIT_MAX_CANDIDATES = 100

    @abstractmethod
    async def max_distance(
        self,
        references: list[np.ndarray],
        other: list[np.ndarray],
        max_candidates: int | None = None,
    ) -> list[list[int]]:
        """
        Get indices of `other` vectors the farthest from every reference vector.

        The methods uses the cosine similarity (not L2 norm) as a metric of distance.

        Args:
            references (list[np.ndarray]): Reference vectors against which all
                distances are calculated.
            other (list[np.ndarray]): List of vectors to which distance from
                the reference vectors are calculated.
            max_candidates (int | None): A maximum number of indices of other vectors
                per reference vector to return. Each reference vector has at most
                `max_candidates` indices of other vectors. If None, defaults to
                the value from the configuration.

        Returns:
            list[list[int]]: For every reference vector, a list of indices of
                vectors being the farthest from it. Each list is ordered by the
                distance to the corresponding reference vector. The farthest
                ones are first. The outer list is aligned with `references`.
        """

    def _validate_arguments(
        self,
        references: Sequence[np.ndarray],
        other: Sequence[np.ndarray],
        max_candidates: int | None = None,
    ) -> list[list[int]] | None:
        if max_candidates is None:
            max_candidates = self.HARD_LIMIT_MAX_CANDIDATES
        if max_candidates <= 0:
            raise ValueError("max_candidates must be greater than zero")

        # An abstract class does not have an initialiser, so it is defined here.
        # pylint: disable=attribute-defined-outside-init
        self.max_candidates = min(max_candidates, self.HARD_LIMIT_MAX_CANDIDATES)
        # pylint: enable=attribute-defined-outside-init

        if not other:
            return [[] for _ in references]
        if not references:
            return []

        return None

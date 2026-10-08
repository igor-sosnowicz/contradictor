"""Module with a custom vector search engine."""

from collections.abc import Sequence
from typing import override

import numpy as np

from src.vector_search.vector_search import VectorSearch


class CustomVectorSearch(VectorSearch):
    """The custom NumPy-based in-memory vector search engine."""

    @override
    async def max_distance(
        self,
        references: Sequence[np.ndarray],
        other: Sequence[np.ndarray],
        max_candidates: int | None = None,
    ) -> list[list[int]]:
        output = self._validate_arguments(
            references=references,
            other=other,
            max_candidates=max_candidates,
        )
        if output is not None:
            return output

        # `copy=True` to avoid modifying parameters.
        reference_matrix = np.asarray(references, dtype=np.float32, copy=True)
        other_matrix = np.asarray(other, dtype=np.float32, copy=True)

        matrix_dimensions = 2
        if (
            reference_matrix.ndim != matrix_dimensions
            or other_matrix.ndim != matrix_dimensions
        ):
            raise ValueError("Vectors must be two-dimensional matrices.")

        if reference_matrix.shape[1] != other_matrix.shape[1]:
            raise ValueError("All vectors must have the same dimensions.")

        reference_norms = np.linalg.norm(
            reference_matrix,
            axis=1,
            keepdims=True,
        )
        other_norms = np.linalg.norm(
            other_matrix,
            axis=1,
            keepdims=True,
        )

        if np.any(reference_norms == 0) or np.any(other_norms == 0):
            raise ValueError("Cosine similarity is undefined for zero vectors.")

        reference_matrix /= reference_norms
        other_matrix /= other_norms

        # Lower cosine similarity means greater cosine distance.
        similarities = reference_matrix @ other_matrix.T

        # The configured instance limit is always an integer. It cannot exceed
        # the number of available candidate vectors.
        limit = min(self.max_candidates, len(other))

        if limit == len(other):
            indices = np.argsort(similarities, axis=1)[:, :limit]
        else:
            indices = np.argpartition(
                similarities,
                kth=limit - 1,
                axis=1,
            )[:, :limit]

            selected_similarities = np.take_along_axis(
                similarities,
                indices,
                axis=1,
            )

            # Sort the selected candidates by greatest distance first.
            order = np.argsort(selected_similarities, axis=1)
            indices = np.take_along_axis(indices, order, axis=1)

        return indices.astype(np.int64).tolist()

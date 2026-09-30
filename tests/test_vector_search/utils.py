"""Module with utility functions facilitating tests of vector search."""

import numpy as np

from src.data_models.data_models import VectorSearchImplementation
from src.pipeline.factories.vector_search_factory import build_vector_search
from src.vector_search.vector_search import VectorSearch


def get_vector_search_instances() -> list[VectorSearch]:
    """
    Get instance of all vector search implementation available.

    Returns:
        list[VectorSearch]: List of default-initialised implementations of
            vector search.
    """
    return [
        build_vector_search(implementation)
        for implementation in VectorSearchImplementation
    ]


def get_random_vectors(n: int, vector_dims: int, seed: int = 0) -> list[np.ndarray]:
    """
    Get a list of random vectors.

    Args:
        n (int): A number of vectors to get.
        vector_dims (int): A dimensionality of each vector.
        seed (int): A random seed. Defaults to 0.

    Returns:
        list[np.ndarray]: List of vectors initialised with values
            samples from a standard (normal) distribution.
    """
    random_generator = np.random.default_rng(seed=seed)
    return [random_generator.standard_normal(vector_dims) for _ in range(n)]

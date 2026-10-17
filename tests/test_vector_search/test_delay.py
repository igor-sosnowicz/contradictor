"""Module with tests for inference speed (delay) of vectors search engines."""

import time
import warnings

import pytest
from loguru import logger

from src.vector_search.vector_search import VectorSearch
from tests.test_vector_search.utils import (
    get_random_vectors,
    get_vector_search_instances,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("db_instance", get_vector_search_instances())
@pytest.mark.parametrize(
    ("other_vectors", "reference_vectors"),
    [
        (1, 1),
        (10, 10),
        (100, 100),
        (1000, 1000),
    ],
)
async def test_search_time(
    db_instance: VectorSearch, other_vectors: int, reference_vectors: int
) -> None:
    """Test if for a given number of vectors a maximum delay is not exceeded."""
    delay_threshold_ms = 100
    vector_dims = 150

    start = time.perf_counter()
    await db_instance.max_distance(
        other=get_random_vectors(n=other_vectors, vector_dims=vector_dims),
        references=get_random_vectors(n=reference_vectors, vector_dims=vector_dims),
    )
    end = time.perf_counter()

    elapsed_ms = (end - start) * 1000
    message = (
        f"Search took {elapsed_ms:.2f} ms (threshold: {delay_threshold_ms} ms) "
        f"for {db_instance.__class__.__name__} with {reference_vectors} "
        f"reference vectors and {other_vectors} other vectors."
    )
    if elapsed_ms >= delay_threshold_ms:
        warnings.warn(
            message,
            RuntimeWarning,
            stacklevel=1,
        )
    else:
        logger.success(message)

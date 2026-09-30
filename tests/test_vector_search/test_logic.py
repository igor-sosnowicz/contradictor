"""Module with tests validating logic of vector search."""

import numpy as np
import numpy.random
import pytest

from src.vector_search.vector_search import VectorSearch
from tests.test_vector_search.utils import (
    get_random_vectors,
    get_vector_search_instances,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("db_instance", get_vector_search_instances())
async def test_empty_other_yields_preserves_length(db_instance: VectorSearch) -> None:
    """Test if empty `other` preserves length of the results."""
    # Seeded to have reproducible results.
    vectors = 10
    vector_length = 15
    references = [
        np.random.default_rng(seed=0).standard_normal(vector_length)
        for _ in range(vectors)
    ]
    results = await db_instance.max_distance(references=references, other=[])

    # A list of empty lists preserving the length is expected.
    assert isinstance(results, list)
    assert len(results) == vectors


@pytest.mark.asyncio
@pytest.mark.parametrize("db_instance", get_vector_search_instances())
async def test_empty_references_yields_empty_result(db_instance: VectorSearch) -> None:
    """Test if empty `references` vectors yield the empty result."""
    # Seeded to have reproducible results.
    vectors = 10
    vector_length = 15
    other = [
        np.random.default_rng(seed=0).standard_normal(vector_length)
        for _ in range(vectors)
    ]
    results = await db_instance.max_distance(references=[], other=other)

    # The empty list is expected.
    assert isinstance(results, list)
    assert not results


@pytest.mark.asyncio
@pytest.mark.parametrize("db_instance", get_vector_search_instances())
@pytest.mark.parametrize(
    ("references_count", "other_count", "max_candidates", "expected_results"),
    [
        (1, 1, 50, 1),  # A limit does not matter if fewer results than limit.
        (1, 25, 50, 25),
        (100, 1, 50, 1),
        (100, 50, 50, 50),  # A limit equals a number of available results.
        (100, 100, 50, 50),  # A limit is respected if more results is available.
    ],
)
async def test_respecting_max_candidates_parameter(
    db_instance: VectorSearch,
    references_count: int,
    other_count: int,
    max_candidates: int,
    expected_results: int,
) -> None:
    """Test if `max_candidates` parameters is respected."""
    vector_dims = 10
    references = get_random_vectors(n=references_count, vector_dims=vector_dims)
    other = get_random_vectors(n=other_count, vector_dims=vector_dims)
    results = await db_instance.max_distance(
        references=references, other=other, max_candidates=max_candidates
    )
    assert results
    for vector_indices in results:
        assert len(vector_indices) == expected_results


@pytest.mark.asyncio
@pytest.mark.parametrize("db_instance", get_vector_search_instances())
# 1 ref, 1 other
async def test_finding_farthest_among_one_reference_and_one_other_vector(
    db_instance: VectorSearch,
) -> None:
    """Test finding the farthest vector for one reference and other vector."""
    vector_dims = 20
    references = get_random_vectors(
        n=1,
        vector_dims=vector_dims,
    )
    other = get_random_vectors(
        n=1,
        vector_dims=vector_dims,
    )

    results = await db_instance.max_distance(references=references, other=other)
    assert len(results) == 1  # Only one reference vector exists.
    assert len(results[0]) == 1  # It contains one other vector.
    assert results[0][0] == 0  # The index of other vector is 0.


@pytest.mark.asyncio
@pytest.mark.parametrize("db_instance", get_vector_search_instances())
# 1 ref, 1000 other
async def test_finding_farthest_among_one_reference_and_1000_other_vectors(
    db_instance: VectorSearch,
) -> None:
    """Test finding the farthest vectors for one reference and 1000 other vectors."""
    vector_dims = 25
    max_candidates = 5
    farthest_vector_id = 20

    reference_vectors_count = 1
    other_vectors_count = 1000

    references = get_random_vectors(
        n=reference_vectors_count,
        vector_dims=vector_dims,
    )
    other = get_random_vectors(
        n=other_vectors_count,
        vector_dims=vector_dims,
    )

    # Ensure `farthest_vector_id` is actually the farthest.
    other[farthest_vector_id] = -references[0]

    results = await db_instance.max_distance(
        references=references, other=other, max_candidates=max_candidates
    )
    assert len(results) == reference_vectors_count
    assert len(results[0]) == max_candidates, (
        f"A number of candidates ({max_candidates}) is not respected. "
        f"Returned {len(results[0])} indices for the first reference vector."
    )
    assert results[0][0] == farthest_vector_id, (
        f"Expected the vector with id = {farthest_vector_id} to be the farthest, but "
        f"a vector search engine returned id = {results[0][0]}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("db_instance", get_vector_search_instances())
# 1000 ref, 1 other
async def test_finding_farthest_among_1000_reference_and_one_other_vector(
    db_instance: VectorSearch,
) -> None:
    """Test finding the farthest vectors for 1000 references and one other vector."""
    vector_dims = 5
    max_candidates = 10

    reference_vectors_count = 1000
    other_vectors_count = 1

    references = get_random_vectors(n=reference_vectors_count, vector_dims=vector_dims)
    other = get_random_vectors(n=other_vectors_count, vector_dims=vector_dims)

    # To make it the guaranteed farthest for the first reference:
    other[0] = -references[0]

    results = await db_instance.max_distance(
        references=references, other=other, max_candidates=max_candidates
    )

    assert len(results) == reference_vectors_count
    for i in range(reference_vectors_count):
        assert len(results[i]) == 1, (
            f"Expected 1 candidate for reference {i}, got {len(results[i])}"
        )
        assert results[i][0] == 0, (
            f"Reference {i} should have returned the only available vector (id 0)"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("db_instance", get_vector_search_instances())
# 1000 ref, 1000 other
async def test_finding_farthest_among_1000_reference_and_1000_other_vectors(
    db_instance: VectorSearch,
) -> None:
    """Test finding the farthest vectors for 1000 references and 1000 other vectors."""
    vector_dims = 7
    max_candidates = 3

    reference_vectors_count = 1000
    other_vectors_count = 1000

    references = get_random_vectors(n=reference_vectors_count, vector_dims=vector_dims)
    other = get_random_vectors(n=other_vectors_count, vector_dims=vector_dims)

    # Create 1000 pairs of opposites.
    # Each other[i] is guaranteed to be the farthest for references[i].
    for i in range(reference_vectors_count):
        other[i] = -references[i]

    results = await db_instance.max_distance(
        references=references, other=other, max_candidates=max_candidates
    )

    assert len(results) == reference_vectors_count
    for i in range(reference_vectors_count):
        assert len(results[i]) == max_candidates, (
            f"Reference {i}: expected {max_candidates} candidates, "
            f"got {len(results[i])}"
        )
        # The 0-th result for reference `i` must be the vector
        # we negated specifically for it.
        assert results[i][0] == i, (
            f"Reference {i} expected farthest vector to be index {i} "
            f"but got {results[i][0]}"
        )

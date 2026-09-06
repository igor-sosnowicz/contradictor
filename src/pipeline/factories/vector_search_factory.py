"""Module with a vector search factory function."""

from src.configuration import config
from src.data_models.data_models import VectorSearchImplementation
from src.pipeline.vector_search import VectorSearch


def build_vector_search(
    vector_search_implementation: VectorSearchImplementation | None = None,
) -> VectorSearch:
    """
    Construct a ready-made vector search from pre-configured implementations.

    Args:
        vector_search_implementation (VectorSearchImplementation): Choice of a
            pre-configured implementation.

    Returns:
        VectorSearch: An initialised instance of the vector search.

    Raises:
        NotImplementedError: Raised if not implemented variant was requested.
    """
    vector_search_implementation = vector_search_implementation or config.vector_search

    match vector_search_implementation:
        case VectorSearchImplementation.NOT_IMPLEMENTED:
            raise NotImplementedError("No vector search is implemented yet.")

"""Module with a vector search factory function."""

from src.configuration import config
from src.data_models.data_models import VectorSearchImplementation
from src.vector_search.vector_search import VectorSearch


def build_vector_search(
    vector_search_implementation: VectorSearchImplementation | None = None,
    **kwargs,
) -> VectorSearch:
    """
    Construct a ready-made vector search from pre-configured implementations.

    Args:
        vector_search_implementation (VectorSearchImplementation): Choice of a
            pre-configured implementation.
        **kwargs: Keyword arguments to be passed on directly to the constructor of
            of implementation.

    Returns:
        VectorSearch: An initialised instance of the vector search.
    """
    vector_search_implementation = vector_search_implementation or config.vector_search

    match vector_search_implementation:
        case VectorSearchImplementation.NUMPY_BASED:
            # pylint: disable=import-outside-toplevel
            from src.vector_search.custom_vector_search import CustomVectorSearch
            # pylint: enable=import-outside-toplevel

            return CustomVectorSearch(**kwargs)

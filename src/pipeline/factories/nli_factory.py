"""Module with a natural language inference factory function."""

from src.configuration import config
from src.data_models.data_models import NLIImplementation
from src.pipeline.nli import NLI


def build_nli(
    nli_implementation: NLIImplementation | None = None,
) -> NLI:
    """
    Construct a ready-made NLI model from pre-configured implementations.

    Args:
        nli_implementation (NLIImplementation): Choice of a pre-configured
            implementation.

    Returns:
        NLI: An initialised instance of the natural language inference model.

    Raises:
        NotImplementedError: Raised if not implemented variant was requested.
    """
    nli_implementation = nli_implementation or config.nli

    match nli_implementation:
        case NLIImplementation.NOT_IMPLEMENTED:
            raise NotImplementedError("No NLI model is implemented yet.")

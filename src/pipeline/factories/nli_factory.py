"""Module with a natural language inference factory function."""

from src.configuration import config
from src.data_models.data_models import NLIImplementation
from src.nli.nli import NLI
from src.nli.transformers_nli import TransformersNLI


def build_nli(nli_implementation: NLIImplementation | None = None, **kwargs) -> NLI:
    """
    Construct a ready-made NLI model from pre-configured implementations.

    Args:
        nli_implementation (NLIImplementation): Choice of a pre-configured
            implementation. Defaults to the model from the configuration.
        kwargs: Additional arguments passed on directly to constructor of
            an NLI implementation.

    Returns:
        NLI: An initialised instance of the natural language inference model.
    """
    nli_implementation = nli_implementation or config.nli

    match nli_implementation:
        case NLIImplementation.TRANSFORMERS_MODERNBERT:
            return TransformersNLI(**kwargs)

        case _:
            raise ValueError(f"Unsupported NLI implementation: {nli_implementation}")

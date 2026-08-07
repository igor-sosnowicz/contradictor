"""Module with utilities shared across all NLI models."""

from src.data_models.data_models import NLIResult
from src.utils.errors import NLIResultError


def to_nli_enum(value: str) -> NLIResult:
    """
    Convert string to the enum with a NLI result.

    Args:
        value (str): A textual value to be converted into a NLI result.

    Returns:
        NLIResult: Enumeration with an NLI result.

    Raises:
        NLIResultError: Raised if an invalid textual value was provided.
    """
    match value.lower().strip():
        case "contradiction":
            return NLIResult.CONTRADICTION
        case "neutral":
            return NLIResult.NEUTRAL
        case "entailment":
            return NLIResult.ENTAILMENT
        case _:
            raise NLIResultError(f"The model returned invalid label for NLI: {value}")

"""Module with tests for utils shared across NLI models."""

import pytest

from src.nli.utils import NLIResult, to_nli_enum
from src.utils.errors import NLIResultError


@pytest.mark.parametrize(
    ("value", "result"),
    [
        ("neutral", NLIResult.NEUTRAL),
        ("NEUTRAL", NLIResult.NEUTRAL),
        ("Neutral", NLIResult.NEUTRAL),
        ("entailment", NLIResult.ENTAILMENT),
        ("ENTAILMENT", NLIResult.ENTAILMENT),
        ("Entailment", NLIResult.ENTAILMENT),
        ("contradiction", NLIResult.CONTRADICTION),
        ("CONTRADICTION", NLIResult.CONTRADICTION),
        ("Contradiction", NLIResult.CONTRADICTION),
        ("", None),  # None = an exception
        (" ", None),
        ("\n", None),
        ("\t", None),
        ("Disagreement", None),
        ("Contradictor is the best!", None),
    ],
)
def test_to_nli_enum(value: str, result: NLIResult | None) -> None:
    """Test converting a textual value into a NLI result being an enum."""
    if result is None:  # None means an exception should be raised.
        with pytest.raises(NLIResultError):
            to_nli_enum(value)
    else:
        converted = to_nli_enum(value)
        assert converted == result

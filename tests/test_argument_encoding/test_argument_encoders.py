"""Tests for the arguments encoders."""

import pytest
import pytest_asyncio

from src.configuration import config
from src.data_models.data_models import (
    StyledFramedArgument,
)
from tests.test_argument_encoding.utils import (
    get_argument_encoders,
    get_random_styled_and_framed_texts,
)


# The session scope saves a lot of time.
@pytest_asyncio.fixture(scope="session")
async def styled_and_framed_arguments() -> list[StyledFramedArgument]:
    """Fixture providing a shared list of styled and framed arguments."""
    return await get_random_styled_and_framed_texts(n=5)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target_dimensions",
    [None, 5, 100],
)
async def test_encoder_output_matches_expected_dimensions(
    target_dimensions: int | None,
    styled_and_framed_arguments: list[StyledFramedArgument],
) -> None:
    """Test if an output of an encoder gives output of the defined dimensions."""
    encoders = get_argument_encoders(target_dimensions)
    arguments = styled_and_framed_arguments

    for encoder in encoders:
        encoder.reset()
        await encoder.prepare()
        output = await encoder(arguments=arguments)

        # None default to the value from the global configuration.
        if target_dimensions is None:
            target_dimensions = config.encoder_output_dimensions

        # Each representation vector has to conform.
        for representation in output:
            assert representation.shape[0] == target_dimensions, (
                f"At least one output of {encoder.__name__} has an invalid "
                f"{representation.shape[0]} outputs, expected {representation.shape[0]}"
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("target_dimensions", [-100, 0])
async def test_encoder_output_for_invalid_output_shape(
    target_dimensions: int | None,
) -> None:
    """Test if an output of an encoder gives output of the defined dimensions."""
    # Only representations of strictly positive dimension count are allowed.
    if target_dimensions is not None and target_dimensions <= 0:
        with pytest.raises(ValueError, match="output dimensions"):
            get_argument_encoders(output_dimensions=target_dimensions)
        return

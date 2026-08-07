"""Module for testing the application entrypoint function."""

import pytest

from src.main import main


@pytest.mark.xfail(reason="Not all components implemented.")
@pytest.mark.asyncio
async def test_main() -> None:
    """Test running the entrypoint function."""
    await main()

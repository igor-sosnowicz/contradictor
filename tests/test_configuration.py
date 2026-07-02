"""Module for testing configuration."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.configuration import Configuration, load_configuration

FAKE_CONFIGURATION_PATH = "./random_dir123"


@pytest.fixture
def invalid_configuration() -> Generator[Path]:
    """Fixture for loading invalid configuration file."""
    content = 'non-existent-key = "wr0ng value3"\n'
    with tempfile.NamedTemporaryFile(mode="r+", suffix=".toml") as file:
        file.write(content)
        file.flush()
        yield Path(file.name)


@pytest.fixture
def valid_configuration() -> Generator[Path]:
    """Fixture for loading invalid configuration file."""
    content = f'data_directory = "{FAKE_CONFIGURATION_PATH}"\n'
    with tempfile.NamedTemporaryFile(mode="r+", suffix=".toml") as file:
        file.write(content)
        file.flush()
        yield Path(file.name)


def test_loading_invalid_configuration(invalid_configuration: Path) -> None:
    """Test if a valid configuration is loaded properly."""
    with pytest.raises(ValidationError, match="non-existent-key"):
        load_configuration(invalid_configuration)


def test_loading_valid_configuration(valid_configuration: Path) -> None:
    """Test if an invalid configuration fails to load."""
    config = load_configuration(valid_configuration)
    assert config
    assert isinstance(config, Configuration)
    assert config.data_directory
    assert isinstance(config.data_directory, Path)
    assert config.data_directory.resolve() == Path(FAKE_CONFIGURATION_PATH).resolve()

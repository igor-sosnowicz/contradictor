"""Module for testing configuration."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from src.configuration import Configuration, load_configuration
from src.utils.errors import ConfigurationError

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
    """Fixture for a valid configuration file."""
    content = f'data_directory = "{FAKE_CONFIGURATION_PATH}"\n'
    with tempfile.NamedTemporaryFile(mode="r+", suffix=".toml") as file:
        file.write(content)
        file.flush()
        yield Path(file.name)


def test_loading_invalid_configuration(invalid_configuration: Path) -> None:
    """Test if an invalid configuration fails to load."""
    with pytest.raises(ConfigurationError, match="non-existent-key"):
        load_configuration(invalid_configuration)


def test_loading_valid_configuration(valid_configuration: Path) -> None:
    """Test if a valid configuration is loaded correctly."""
    config = load_configuration(valid_configuration)
    assert isinstance(config, Configuration)
    assert isinstance(config.data_directory, Path)
    assert config.data_directory.resolve() == Path(FAKE_CONFIGURATION_PATH).resolve()


def test_loading_missing_configuration() -> None:
    """Test if loading a non-existent configuration file does not raise."""
    with pytest.raises(ConfigurationError, match="file is missing"):
        load_configuration(Path("/non/existent/path"))

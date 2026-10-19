"""Module for testing configuration."""

import tempfile
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.configuration
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
    """Test if an explicitly given, non-existent configuration file raises."""
    with pytest.raises(ConfigurationError, match="file is missing"):
        load_configuration(Path("/non/existent/path"))


def test_loading_no_file_returns_defaults() -> None:
    """Test if the absence of a configuration file yields pure defaults."""
    config = load_configuration(None)
    assert isinstance(config, Configuration)
    assert config.data_directory == src.configuration.directories.user_data_path
    assert config.cache_directory == src.configuration.directories.user_cache_path


def test_resolve_configuration_file_without_any_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test if the resolver finds no candidates and yields None (defaults)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "src.configuration.directories",
        SimpleNamespace(user_config_path=tmp_path / "xdg" / "Contradictor"),
    )
    assert src.configuration._resolve_configuration_file() is None  # pylint: disable=protected-access


def test_resolve_configuration_file_prefers_dev_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test if `./config.toml` wins over the platformdirs candidate."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "src.configuration.directories",
        SimpleNamespace(user_config_path=tmp_path / "xdg" / "Contradictor"),
    )
    (tmp_path / "config.toml").write_text("", encoding="utf-8")
    assert src.configuration._resolve_configuration_file() == Path(  # pylint: disable=protected-access
        "./config.toml"
    )


def test_resolve_configuration_file_falls_back_to_platformdirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test if the platformdirs candidate is used when the dev file is absent."""
    xdg_config = tmp_path / "xdg" / "Contradictor"
    xdg_config.mkdir(parents=True)
    (xdg_config / "config.toml").write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "src.configuration.directories",
        SimpleNamespace(user_config_path=xdg_config),
    )
    assert (  # pylint: disable=protected-access
        src.configuration._resolve_configuration_file() == xdg_config / "config.toml"
    )

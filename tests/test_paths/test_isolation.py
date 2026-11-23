"""Test-isolation checks: the conftest redirect keeps tests off the real machine."""

from uuid import uuid4

import pytest
from platformdirs import PlatformDirs

from src.paths import paths


def test_roots_live_under_the_pytest_tmp_tree(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """The autouse conftest fixture points all three roots into the pytest tmp tree."""
    basetemp = tmp_path_factory.getbasetemp()

    for root in (paths.data_root, paths.cache_root, paths.tmp_root):
        assert root.is_relative_to(basetemp)


def test_resolved_paths_land_under_the_redirected_roots() -> None:
    """A resolved path is created inside the redirected tree, not the real one."""
    resolved = paths.get_data_path(
        "isolation/probe.json", target_module="isolation", is_file=True
    )

    assert resolved.is_relative_to(paths.data_root)
    assert resolved.parent.is_dir()


def test_nothing_is_written_to_the_real_user_directories() -> None:
    """A probe grant must not appear under the production platformdirs roots."""
    probe = f"ups-v2-probe-{uuid4().hex}"
    resolved = paths.get_data_path(
        f"{probe}.json", target_module="isolation", is_file=True
    )

    directories = PlatformDirs("Contradictor")
    real_data = directories.user_data_path
    real_entries = {path.name for path in real_data.glob("*")}

    assert not (real_data / "isolation" / f"{probe}.json").exists()
    assert resolved.name not in real_entries


def test_probe_never_reaches_the_real_cache_root() -> None:
    """The cache and tmp roots are redirected with the same guarantee."""
    probe = f"ups-v2-probe-{uuid4().hex}"
    tmp_resolved = paths.get_tmp_path(f"{probe}.bin", target_module="isolation")

    directories = PlatformDirs("Contradictor")
    real_cache = directories.user_cache_path
    real_entries = {path.name for path in real_cache.glob("*")}

    assert tmp_resolved.is_relative_to(paths.tmp_root)
    assert f"{probe}.bin" not in real_entries
    assert not (real_cache / probe).exists()

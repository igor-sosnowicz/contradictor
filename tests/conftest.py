"""Shared fixtures for the whole test suite."""

from collections.abc import Iterator

import pytest

from src.paths import initialise, paths


@pytest.fixture(autouse=True, scope="module")
def isolated_path_roots(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """
    Redirect all three path roots into a per-module pytest temporary directory.

    Invariant 2 of the Unified Path System v2: tests must never touch the
    real machine. This is the only place roots are overridden; the
    previous roots are restored afterwards.
    """
    roots = tmp_path_factory.mktemp("path-roots")
    saved = (paths.data_root, paths.cache_root, paths.tmp_root)
    initialise(
        data_root=roots / "data",
        cache_root=roots / "cache",
        tmp_root=roots / "tmp",
    )
    yield
    initialise(data_root=saved[0], cache_root=saved[1], tmp_root=saved[2])

"""Per-test fixtures giving every router test virgin roots and an empty ledger."""

from collections.abc import Iterator

import pytest

from src.paths import initialise, paths


@pytest.fixture(autouse=True)
def fresh_router(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """
    Redirect all three roots into a per-test pytest temporary directory.

    Mirrors the module-scoped fixture in `tests/conftest.py`, but runs per
    test: `initialise()` to fresh roots also clears the usage ledger, so
    every test starts with virgin roots and no recorded grants or claims.
    """
    roots = tmp_path_factory.mktemp("router-test-roots")
    saved = (paths.data_root, paths.cache_root, paths.tmp_root)
    initialise(
        data_root=roots / "data",
        cache_root=roots / "cache",
        tmp_root=roots / "tmp",
    )
    yield
    initialise(data_root=saved[0], cache_root=saved[1], tmp_root=saved[2])

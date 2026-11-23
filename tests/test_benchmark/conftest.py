"""Per-test fixtures giving benchmark tests virgin path roots."""

from collections.abc import Iterator

import pytest

from src.paths import initialise, paths


@pytest.fixture(autouse=True)
def fresh_benchmark_roots(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """
    Redirect all three roots into a per-test pytest temporary directory.

    The module-scoped fixture in `tests/conftest.py` redirects roots for the
    whole module; benchmark tests count files in the results directory, so
    each test needs its own virgin roots to see exactly its own output.
    """
    roots = tmp_path_factory.mktemp("benchmark-test-roots")
    saved = (paths.data_root, paths.cache_root, paths.tmp_root)
    initialise(
        data_root=roots / "data",
        cache_root=roots / "cache",
        tmp_root=roots / "tmp",
    )
    yield
    initialise(data_root=saved[0], cache_root=saved[1], tmp_root=saved[2])

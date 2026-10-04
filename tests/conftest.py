"""Shared fixtures isolating the unified path registry during tests."""

from collections.abc import Iterator

import pytest

from src.paths import registry


@pytest.fixture(autouse=True, scope="module")
def isolated_path_roots(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[None]:
    """
    Redirect the path registry roots into a temporary directory for each module.

    Without this, any component resolving a path while it is constructed would
    create directories under the real user data and cache roots, leaking state
    between tests and polluting the machine running the suite.

    The scope must stay at module level: some integration fixtures are declared
    with `scope="module"`, and pytest builds those before any function-scoped
    fixture, so a function-scoped override would be applied too late.

    Args:
        tmp_path_factory (pytest.TempPathFactory): Factory for per-module
            temporary directories.

    Yields:
        None: While the registry points at temporary roots.
    """
    root = tmp_path_factory.mktemp("path_roots")
    previous = registry._roots
    registry.initialise(
        data_root=root / "data",
        cache_root=root / "cache",
    )
    try:
        yield
    finally:
        registry._roots = previous

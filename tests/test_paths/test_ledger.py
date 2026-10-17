"""Usage-ledger tests: contents, clearing on root change, no-op `initialise`."""

from src.paths import initialise, paths
from src.paths.core import PathRoot


def test_ledger_records_grants_in_order() -> None:
    """Grants appear in grant order with their root, slice and forced flag."""
    cache = paths.get_cache_path("metrics.parquet", target_module="search_module")
    data = paths.get_data_path(target_module="search_module")

    grants = paths.ledger

    assert [grant.resolved_path for grant in grants] == [cache, data]
    assert grants[0].root_kind is PathRoot.CACHE
    assert grants[0].module_slice == "search_module"
    assert grants[0].resolved_path == cache
    assert grants[0].forced is False
    assert grants[1].root_kind is PathRoot.DATA
    assert grants[1].module_slice == "search_module"
    assert grants[1].resolved_path == data


def test_dynamic_helper_grants_carry_the_shared_slice() -> None:
    """Shared-dir exception: dynamic-helper grants are recorded under `shared`."""
    resolved = paths.raw_dataset_path("persuade")

    grant = paths.ledger[-1]

    assert grant.root_kind is PathRoot.DATA
    assert grant.module_slice == "shared"
    assert grant.resolved_path == resolved
    assert grant.forced is False


def test_ledger_is_cleared_when_a_root_changes() -> None:
    """Outstanding grants refer to the previous roots, so a change clears them."""
    paths.get_data_path("probe.json", target_module="search_module")
    assert paths.ledger

    initialise(cache_root=paths.cache_root / "moved")

    assert paths.ledger == ()


def test_noop_initialise_keeps_the_ledger() -> None:
    """Calling `initialise()` without arguments changes no root and clears nothing."""
    paths.get_data_path("probe.json", target_module="search_module")
    before = paths.ledger

    initialise()

    assert paths.ledger == before


def test_reinitialising_the_same_roots_keeps_the_ledger() -> None:
    """Only an actual root *change* clears the ledger, not a restatement."""
    paths.get_data_path("probe.json", target_module="search_module")
    before = paths.ledger

    initialise(
        data_root=paths.data_root,
        cache_root=paths.cache_root,
        tmp_root=paths.tmp_root,
    )

    assert paths.ledger == before

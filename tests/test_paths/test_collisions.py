"""Collision-rule tests: cross-claim raises, `force` overrides, `shared` semantics."""

import pytest

from src.paths import paths

SHARED = "shared"


def test_slice_claim_colliding_with_shared_raises() -> None:
    """A module-slice grant colliding with a `shared` claim raises (spec Q7)."""
    paths.model_path("classifier.pkl")

    with pytest.raises(ValueError, match="already claimed by module slice 'shared'"):
        paths.get_data_path("", target_module="models")


def test_shared_claim_colliding_with_slice_raises() -> None:
    """The reverse order collides too: `shared` first, module slice second."""
    paths.get_data_path("", target_module="models")

    with pytest.raises(ValueError, match="already claimed by module slice 'models'"):
        paths.model_path("classifier.pkl")


def test_force_overrides_a_collision_and_is_recorded() -> None:
    """`force=True` allows the override and the grant is recorded as forced."""
    paths.get_data_path("", target_module="models")

    resolved = paths.model_path("classifier.pkl", force=True)

    assert resolved == paths.data_root / "models" / "classifier.pkl"
    grant = paths.ledger[-1]
    assert grant.module_slice == SHARED
    assert grant.resolved_path == resolved
    assert grant.forced is True


def test_force_on_a_sliced_override_is_recorded_with_the_slice() -> None:
    """A forced sliced grant keeps the claiming slice in its ledger record."""
    paths.model_path("classifier.pkl")

    resolved = paths.get_data_path("", target_module="models", force=True)

    assert resolved == paths.data_root / "models"
    grant = paths.ledger[-1]
    assert grant.module_slice == "models"
    assert grant.forced is True


def test_same_slice_regrant_does_not_raise() -> None:
    """Re-claiming your own slice never raises (spec Q7)."""
    first = paths.get_data_path("metrics.parquet", target_module="search_module")
    second = paths.get_data_path("metrics.parquet", target_module="search_module")

    assert first == second
    assert all(grant.module_slice == "search_module" for grant in paths.ledger)
    assert all(not grant.forced for grant in paths.ledger)


def test_shared_regrant_does_not_raise() -> None:
    """
    Dynamic helpers from different callers share their root directories.

    Amendment A1 (spec §3): the helpers claim as `shared` regardless of the
    calling module, so the production pipeline (framer + extractors in one
    process) never trips the collision guard.
    """
    synthetic_framer = {"__file__": "/synthetic/framer.py", "paths": paths}
    synthetic_extractor = {"__file__": "/synthetic/extractor.py", "paths": paths}

    exec("resolved = paths.model_path('classifier.pkl')", synthetic_framer)  # noqa: S102
    exec("resolved = paths.model_path('classifier.pkl')", synthetic_extractor)  # noqa: S102

    framer_path = synthetic_framer["resolved"]
    extractor_path = synthetic_extractor["resolved"]
    assert framer_path == extractor_path
    assert framer_path == paths.data_root / "models" / "classifier.pkl"
    assert all(grant.module_slice == SHARED for grant in paths.ledger)


def test_distinct_dynamic_helpers_under_one_shared_directory_do_not_raise() -> None:
    """File claims on the same shared parent coexist under the `shared` slice."""
    model = paths.model_path("classifier.pkl")
    vectorizer = paths.model_path("vectorizer.pkl")

    assert model.parent == vectorizer.parent == paths.data_root / "models"
    assert [grant.resolved_path for grant in paths.ledger] == [model, vectorizer]


def test_distinct_shared_subdirectories_do_not_collide() -> None:
    """Different shared subdirectories are independent claims."""
    raw = paths.raw_dataset_path("persuade")
    processed = paths.processed_dataset_path("persuade")

    assert raw == paths.data_root / "raw_datasets" / "persuade"
    assert processed == paths.data_root / "processed_dataset" / "persuade"


def test_dynamic_dir_name_claims_the_directory_itself() -> None:
    """A dynamic name without a suffix claims the named directory, not its parent."""
    resolved = paths.raw_dataset_path("mp_frame_corpus")

    assert resolved == paths.data_root / "raw_datasets" / "mp_frame_corpus"
    assert resolved.is_dir()
    # A second shared claim on the same directory must not raise.
    assert paths.raw_dataset_path("mp_frame_corpus") == resolved


def test_cross_slice_sliced_getters_never_collide() -> None:
    """Different module slices resolve to different directories by construction."""
    first = paths.get_cache_path("shared.bin", target_module="slice_a")
    second = paths.get_cache_path("shared.bin", target_module="slice_b")

    assert first.parent != second.parent
    assert not any(grant.forced for grant in paths.ledger)


@pytest.mark.parametrize("helper", ["cache_path"])
def test_cache_helper_claims_the_cache_root_for_files(helper: str) -> None:
    """`cache_path` with a file name claims the cache root itself as `shared`."""
    resolved = getattr(paths, helper)("probe.bin")

    assert resolved == paths.cache_root / "probe.bin"
    assert paths.ledger[-1].module_slice == SHARED

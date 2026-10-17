"""Guard and sanitisation tests for subpaths, dynamic names and target_module."""

import pytest

from src.paths import paths

INSIDE_ROOT = "must stay inside its root"
FORWARD_SLASH = "must use '/'"
NO_TRAILING = "must not end with"
NO_EMPTY_SEGMENTS = "must not contain empty"
SINGLE_SEGMENT = "single path segment"


@pytest.mark.parametrize(
    ("subpath", "reason"),
    [
        pytest.param("/etc/cron.d/x", INSIDE_ROOT, id="absolute"),
        pytest.param("C:/Windows", INSIDE_ROOT, id="windows-drive"),
        pytest.param("..", INSIDE_ROOT, id="parent"),
        pytest.param("../escape", INSIDE_ROOT, id="parent-prefix"),
        pytest.param("a\\b", FORWARD_SLASH, id="backslash"),
        pytest.param("model.", NO_TRAILING, id="trailing-dot"),
        pytest.param("model ", NO_TRAILING, id="trailing-space"),
        pytest.param("sub//out.json", NO_EMPTY_SEGMENTS, id="empty-segment"),
        pytest.param("sub/./out.json", NO_EMPTY_SEGMENTS, id="dot-segment"),
    ],
)
def test_getters_reject_non_portable_subpaths(subpath: str, reason: str) -> None:
    """Every sliced getter passes its subpath through the portability guards."""
    with pytest.raises(ValueError, match=reason):
        paths.get_data_path(subpath, target_module="search_module")


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        pytest.param("", SINGLE_SEGMENT, id="empty"),
        pytest.param(".", SINGLE_SEGMENT, id="dot"),
        pytest.param("..", INSIDE_ROOT, id="parent"),
        pytest.param("../escape", INSIDE_ROOT, id="parent-prefix"),
        pytest.param("a/b", SINGLE_SEGMENT, id="extra-segment"),
        pytest.param("a\\b", FORWARD_SLASH, id="backslash"),
        pytest.param("C:/Windows", INSIDE_ROOT, id="windows-drive"),
        pytest.param("model.", NO_TRAILING, id="trailing-dot"),
    ],
)
def test_dynamic_names_must_be_single_portable_segments(name: str, reason: str) -> None:
    """A dynamic name is sanitised, then held to exactly one portable segment."""
    with pytest.raises(ValueError, match=reason):
        paths.model_path(name)


@pytest.mark.parametrize(
    ("raw", "sanitised"),
    [
        pytest.param("argument-framing", "argument_framing", id="hyphen"),
        pytest.param("sbert fine-tune", "sbert_fine_tune", id="space-and-hyphen"),
    ],
)
def test_dynamic_names_sanitise_spaces_and_hyphens(raw: str, sanitised: str) -> None:
    """The documented normalisation must keep working for real dataset names."""
    resolved = paths.raw_dataset_path(raw)

    assert resolved == paths.data_root / "raw_datasets" / sanitised
    assert resolved.is_dir()


@pytest.mark.parametrize(
    ("target_module", "reason"),
    [
        pytest.param("a/b", SINGLE_SEGMENT, id="extra-segment"),
        pytest.param("", SINGLE_SEGMENT, id="empty"),
        pytest.param(".", SINGLE_SEGMENT, id="dot"),
        pytest.param("..", INSIDE_ROOT, id="parent"),
        pytest.param("a\\b", FORWARD_SLASH, id="backslash"),
        pytest.param("slice.", NO_TRAILING, id="trailing-dot"),
    ],
)
def test_target_module_must_be_a_single_portable_segment(
    target_module: str,
    reason: str,
) -> None:
    """The cross-module override is guarded like a dynamic name."""
    with pytest.raises(ValueError, match=reason):
        paths.get_cache_path(target_module=target_module)


def test_dynamic_names_do_not_leak_into_slices() -> None:
    """A sanitised dynamic name resolves under the shared directory, not a slice."""
    resolved = paths.model_path("frame classifier")

    assert resolved == paths.data_root / "models" / "frame_classifier"

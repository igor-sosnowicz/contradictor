"""Regression tests for path containment and root handling in the registry."""

import subprocess
import sys
from enum import Enum
from pathlib import Path

import pytest

from src.paths import (
    cache_path,
    ensure_all,
    initialise,
    model_path,
    processed_dataset_path,
    raw_dataset_path,
    registry,
    resolve,
)
from src.paths.core import CorePaths, PathRoot, PathSpec

REPO_ROOT = Path(__file__).resolve().parents[2]

DYNAMIC_HELPERS = (
    raw_dataset_path,
    processed_dataset_path,
    model_path,
    cache_path,
)


@pytest.mark.parametrize("helper", DYNAMIC_HELPERS)
@pytest.mark.parametrize(
    "name",
    [
        pytest.param("", id="empty"),
        pytest.param(".", id="dot"),
        pytest.param("..", id="parent"),
        pytest.param("../escape", id="parent-prefix"),
        pytest.param("../../etc", id="traversal"),
        pytest.param("/etc/cron.d/x", id="absolute"),
        pytest.param("a/b", id="extra-segment"),
        pytest.param("a\\b", id="windows-separator"),
        pytest.param("C:/Windows", id="windows-drive"),
        pytest.param("C:\\Windows", id="windows-drive-backslash"),
        pytest.param("..\\..\\Windows", id="windows-traversal"),
        pytest.param("model.", id="trailing-dot"),
        pytest.param("model ", id="trailing-space"),
    ],
)
def test_dynamic_helper_rejects_names_that_leave_the_parent(
    helper: object,
    name: str,
) -> None:
    """A dynamic name must never resolve outside its declared parent directory."""
    with pytest.raises(ValueError, match="single path segment"):
        helper(name)  # type: ignore[operator]


@pytest.mark.parametrize("helper", DYNAMIC_HELPERS)
def test_dynamic_helper_sanitises_spaces_and_hyphens(helper: object) -> None:
    """The documented normalisation must keep working for real dataset names."""
    resolved = helper("argument-framing")  # type: ignore[operator]
    assert resolved.name == "argument_framing"


@pytest.mark.parametrize(
    "relative",
    [
        pytest.param("..", id="parent"),
        pytest.param("../escape", id="parent-prefix"),
        pytest.param("/etc", id="absolute"),
        pytest.param("C:/Windows", id="windows-drive"),
        pytest.param("C:\\Windows", id="windows-drive-backslash"),
        pytest.param("..\\..\\Windows", id="windows-traversal"),
        pytest.param("\\\\server\\share\\x", id="unc"),
    ],
)
def test_path_spec_rejects_declarations_that_leave_the_root(relative: str) -> None:
    """Both POSIX and Windows syntax must be rejected, not just POSIX."""
    with pytest.raises(ValueError, match="stay inside its root"):
        PathSpec(PathRoot.DATA, relative, is_dir=True)


def test_path_spec_rejects_an_empty_declaration() -> None:
    """An empty `relative` is a bug, and is ambiguous with `"."`."""
    with pytest.raises(ValueError, match="must not be empty"):
        PathSpec(PathRoot.DATA, "", is_dir=True)


@pytest.mark.parametrize(
    ("relative", "reason"),
    [
        pytest.param("a\\b", "must use '/'", id="windows-separator"),
        pytest.param("sub\\dir", "must use '/'", id="windows-separator-nested"),
        pytest.param("model.", "must not end with", id="trailing-dot"),
        pytest.param("model ", "must not end with", id="trailing-space"),
    ],
)
def test_path_spec_rejects_non_portable_spellings(relative: str, reason: str) -> None:
    """
    Reject names that would denote a different tree on Windows than on POSIX.

    A backslash nests on Windows but is an ordinary filename character on POSIX,
    and Windows strips a trailing dot or space, so both spellings would silently
    diverge between the team's platforms.
    """
    with pytest.raises(ValueError, match=reason):
        PathSpec(PathRoot.DATA, relative, is_dir=True)


def test_path_spec_rejects_a_non_portable_chained_segment() -> None:
    """Chaining must not reintroduce a spelling that varies by platform."""
    base = PathSpec(PathRoot.DATA, "models", is_dir=True)
    with pytest.raises(ValueError, match="must use '/'"):
        base / "a\\b"


def test_path_spec_rejects_a_file_spec_naming_no_location() -> None:
    """A file spec of `.` would make `ensure_all` create the root's parent."""
    with pytest.raises(ValueError, match="must name a location"):
        PathSpec(PathRoot.DATA, ".", is_dir=False)


def test_path_spec_allows_a_dir_spec_naming_the_root() -> None:
    """`CACHES_DIR` declares the cache root itself as `.` and must stay legal."""
    assert CorePaths.CACHES_DIR.value.relative == "."
    assert CorePaths.CACHES_DIR.value.is_dir


@pytest.mark.parametrize(
    "sub",
    [
        pytest.param("", id="empty"),
        pytest.param(".", id="dot"),
    ],
)
def test_chaining_rejects_segments_that_would_alias_the_parent(sub: str) -> None:
    """`PurePosixPath` drops an empty or `.` child, aliasing the parent spec."""
    base = PathSpec(PathRoot.DATA, "models", is_dir=True)
    with pytest.raises(ValueError, match="Cannot chain an empty segment"):
        base / sub


@pytest.mark.parametrize(
    "sub",
    [
        pytest.param("..", id="parent"),
        pytest.param("/etc", id="absolute"),
    ],
)
def test_chaining_rejects_escaping_segments(sub: str) -> None:
    """A chained segment must stay inside the root, as a declaration does."""
    base = PathSpec(PathRoot.DATA, "models", is_dir=True)
    with pytest.raises(ValueError, match="stay inside its root"):
        base / sub


def test_chaining_builds_a_file_child() -> None:
    """The documented chaining form must keep working."""
    child = (PathSpec(PathRoot.DATA, "models", is_dir=True) / "model.pkl").as_file()
    assert child.relative == "models/model.pkl"
    assert not child.is_dir


def test_resolve_routes_on_root_equality_not_identity() -> None:
    """A `PathSpec` built from a plain string must not be routed to the cache."""

    class DataSpec(Enum):
        # A plain string is used deliberately: routing must compare by equality.
        MEMBER = PathSpec("data", "models", is_dir=True)  # type: ignore[arg-type]

    original = registry.ALL_WISHLISTS
    registry.ALL_WISHLISTS = (*original, DataSpec)
    try:
        roots = registry._active_roots()
        assert resolve(DataSpec.MEMBER) == roots.data / "models"
    finally:
        registry.ALL_WISHLISTS = original


def test_resolve_rejects_an_unregistered_wishlist() -> None:
    """An unregistered member resolves today but is never created by `ensure_all`."""

    class Unregistered(Enum):
        MEMBER = PathSpec(PathRoot.DATA, "models", is_dir=True)

    with pytest.raises(ValueError, match="not a registered wishlist"):
        resolve(Unregistered.MEMBER)


def test_resolve_rejects_a_non_path_spec_value() -> None:
    """A malformed wishlist must fail loudly rather than raise `AttributeError`."""

    class Malformed(Enum):
        MEMBER = "models"

    original = registry.ALL_WISHLISTS
    registry.ALL_WISHLISTS = (*original, Malformed)
    try:
        with pytest.raises(TypeError, match="must hold a PathSpec"):
            resolve(Malformed.MEMBER)
    finally:
        registry.ALL_WISHLISTS = original


def test_initialise_with_both_roots_needs_no_config_file(tmp_path: Path) -> None:
    """Passing both roots must not read `config.toml`, so cwd cannot break tests."""
    script = (
        "import sys; sys.path.insert(0, {repo!r})\n"
        "from pathlib import Path\n"
        "from src.paths import initialise, raw_dataset_path\n"
        "initialise(data_root=Path({data!r}), cache_root=Path({cache!r}))\n"
        "print(raw_dataset_path('persuade'))\n"
    ).format(
        repo=str(REPO_ROOT),
        data=str(tmp_path / "data"),
        cache=str(tmp_path / "cache"),
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(tmp_path / "data" / "raw_datasets" / "persuade")


def test_initialise_keeps_previous_roots_when_creation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed `mkdir` must not leave the registry pointing at a partial tree."""
    previous = registry._roots
    original_mkdir = Path.mkdir
    calls = {"count": 0}
    allowed_before_failure = 2

    def flaky_mkdir(self: Path, *args: object, **kwargs: object) -> None:
        calls["count"] += 1
        if calls["count"] > allowed_before_failure:
            message = "simulated failure"
            raise OSError(message)
        original_mkdir(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "mkdir", flaky_mkdir)
    with pytest.raises(OSError, match="simulated failure"):
        initialise(data_root=tmp_path / "d2", cache_root=tmp_path / "c2")
    assert registry._roots == previous


def test_initialise_without_arguments_is_idempotent(tmp_path: Path) -> None:
    """Calling with no arguments keeps the active roots and only creates gaps."""
    initialise(data_root=tmp_path / "data", cache_root=tmp_path / "cache")
    first = ensure_all()
    second = initialise()
    assert first == second
    assert all(path.is_dir() for path in second)

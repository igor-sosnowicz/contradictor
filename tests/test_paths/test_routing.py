"""Routing-shape and slice-derivation tests for the v2 PathRouter."""

import inspect
import subprocess
import sys
import types
from pathlib import Path

import pytest

from src.paths import paths
from src.paths.router import INTERACTIVE_SLICE, PathRouter

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("getter_name", "root_property"),
    [
        ("get_data_path", "data_root"),
        ("get_cache_path", "cache_root"),
        ("get_tmp_path", "tmp_root"),
    ],
)
def test_routing_shape_is_root_slice_subpath(
    getter_name: str,
    root_property: str,
) -> None:
    """Every getter must resolve to `<root>/<module_slice>/<subpath>`."""
    resolved = getattr(paths, getter_name)(
        "metrics.parquet", target_module="search_module"
    )
    root = getattr(paths, root_property)

    assert resolved == root / "search_module" / "metrics.parquet"
    assert resolved.parent.is_dir()


def test_caller_slice_comes_from_the_repo_relative_path() -> None:
    """
    A call from a real repo file derives its slice by stack inspection.

    This test file lives at `tests/test_paths/…`, so the first repo-relative
    component — the slice — is `tests`.
    """
    resolved = paths.get_cache_path("probe.json", is_file=True)

    assert resolved == paths.cache_root / "tests" / "probe.json"


def test_target_module_overrides_the_caller_slice() -> None:
    """An explicit `target_module` wins over stack inspection."""
    resolved = paths.get_cache_path("checkpoint.pt", target_module="argument_detection")

    assert resolved == paths.cache_root / "argument_detection" / "checkpoint.pt"


def test_slice_is_the_first_component_after_src() -> None:
    """A `src` module's canonical id encodes its module slice."""
    assert PathRouter._slice_from_canonical_id("search_module/cache") == "search_module"


def test_slice_of_a_top_level_file_is_its_stem() -> None:
    """A top-level file (`main.py`) uses its stem as the slice."""
    assert PathRouter._slice_from_canonical_id("main.py") == "main"


def test_slice_of_an_empty_canonical_id_is_interactive() -> None:
    """An empty canonical id cannot name a slice."""
    assert PathRouter._slice_from_canonical_id("") == INTERACTIVE_SLICE


def test_synthetic_frame_falls_back_to_interactive() -> None:
    """A frame whose `__file__` is not a real file attributes to `interactive`."""

    def caller() -> Path:
        return paths.get_tmp_path("probe.json", is_file=True)

    synthetic_globals = {"__file__": "/nonexistent/synthetic_frame.py", "paths": paths}
    synthetic_caller = types.FunctionType(caller.__code__, synthetic_globals)

    resolved = synthetic_caller()

    assert resolved == paths.tmp_root / INTERACTIVE_SLICE / "probe.json"
    assert resolved.parent.is_dir()


def test_frames_outside_the_repo_are_skipped() -> None:
    """
    A real file outside the repo cannot anchor, so it is skipped.

    The first usable frame is then this test file — slice `tests` — instead
    of a `ValueError` leaking out of canonical-id resolution. The stdlib
    file is used because in-repo tooling (`.venv/…`) *is* inside the repo
    and would map to its own slice.
    """

    def caller() -> Path:
        return paths.get_data_path("probe.json", is_file=True)

    synthetic_globals = {"__file__": inspect.__file__, "paths": paths}
    synthetic_caller = types.FunctionType(caller.__code__, synthetic_globals)

    resolved = synthetic_caller()

    assert resolved == paths.data_root / "tests" / "probe.json"


def test_python_dash_c_caller_falls_back_to_interactive(tmp_path: Path) -> None:
    """
    A `python -c` caller has no file context and resolves the `interactive` slice.

    All three roots are injected via `initialise()`, so the script never
    reads a configuration file and the CWD cannot influence the result.
    """
    data_root, cache_root, tmp_root = tmp_path / "d", tmp_path / "c", tmp_path / "t"
    script = f"""\
import sys; sys.path.insert(0, {str(REPO_ROOT)!r})
from pathlib import Path
from src.paths import initialise, paths
initialise(
    data_root=Path({str(data_root)!r}),
    cache_root=Path({str(cache_root)!r}),
    tmp_root=Path({str(tmp_root)!r}),
)
print(paths.get_data_path('probe.json', is_file=True))
"""

    result = subprocess.run(  # noqa: S603 Intended subprocess test, not a security issue.
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(data_root / "interactive" / "probe.json")


def test_subpath_empty_names_the_slice_directory() -> None:
    """`subpath=""` resolves to and creates the module slice directory itself."""
    resolved = paths.get_tmp_path(target_module="search_module")

    assert resolved == paths.tmp_root / "search_module"
    assert resolved.is_dir()


def test_file_suffix_creates_only_the_parent_directory() -> None:
    """A final segment with a suffix is a file: only its parent is created."""
    resolved = paths.get_cache_path(
        "sub/out.json", target_module="search_module", is_file=True
    )

    assert resolved.parent.is_dir()
    assert not resolved.exists()


def test_no_suffix_creates_the_directory_itself() -> None:
    """A final segment without a suffix is a directory: it is created."""
    resolved = paths.get_cache_path("sub/results", target_module="search_module")

    assert resolved.is_dir()

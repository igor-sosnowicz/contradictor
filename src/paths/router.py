"""
PathRouter: the single source of filesystem truth.

The full usage contract (routing rule, slice derivation, guards,
collisions, ledger, root injection) lives in `docs/UPS.md`.
"""

import inspect
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, cast

from src.paths.core import (
    PathRoot,
    portable_segments,
    sanitise_segment,
    single_segment,
)
from src.paths.environment import RepositoryEnvironment
from src.paths.ledger import GrantLedger, PathGrant

if TYPE_CHECKING:
    from src.configuration import Configuration

INTERACTIVE_SLICE = "interactive"
"""Module slice attributed to callers without a file context (REPL, `-c`)."""

SHARED_SLICE = "shared"
"""
Pseudo-slice attributing dynamic-helper claims to shared locations.

Dynamic helpers address shared root-level directories by design
(`<data>/models`, `<data>/raw_datasets/`, …), so their claims belong to no
single module slice: the whole grant — claim and ledger record — is
attributed to `shared`, and the caller's slice is never consulted.
"""


def _configuration() -> "Configuration":
    """
    Return the project configuration - lazy import.

    Imported lazily so that importing `src.paths` never pulls the full
    configuration stack, and so that explicit roots work even before a
    configuration file is read.
    """
    # pylint: disable=import-outside-toplevel
    from src.configuration import config

    return config


class PathRouter:
    """
    Route caller-scoped paths under the data, cache and tmp roots.

    Every resolved path looks like `<root>/<module_slice>/<subpath>`;
    the slice comes from stack inspection or an explicit `target_module`.
    Full contract (guards, collisions, ledger): `docs/UPS.md`.
    """

    # --- Roots ---
    def __init__(self) -> None:
        """Initialise the router with lazily resolved, unclaimed roots."""
        self._data_root: Path | None = None
        self._cache_root: Path | None = None
        self._tmp_root: Path | None = None
        self._root_map: dict[PathRoot, Path] | None = None
        self._config_cache: Configuration | None = None
        self._ledger = GrantLedger()
        self._environment = RepositoryEnvironment()

    def _config(self) -> "Configuration":
        """Project configuration, cached on the router after first use."""
        if self._config_cache is None:
            self._config_cache = _configuration()
        return self._config_cache

    def _resolve_unset_roots(self) -> None:
        """
        Populate every still-unset root from the configuration.

        Single configuration touch: one access resolves data, cache and
        tmp together. Roots injected via `initialise()` are never overwritten.
        """
        config = self._config()
        self._data_root = self._data_root or config.data_directory
        self._cache_root = self._cache_root or config.cache_directory
        self._tmp_root = self._tmp_root or config.tmp_directory

    @property
    def data_root(self) -> Path:
        """Data root, resolved lazily from the configuration."""
        if self._data_root is None:
            self._resolve_unset_roots()
        return cast("Path", self._data_root)

    @property
    def cache_root(self) -> Path:
        """Cache root, resolved lazily from the configuration."""
        if self._cache_root is None:
            self._resolve_unset_roots()
        return cast("Path", self._cache_root)

    @property
    def tmp_root(self) -> Path:
        """Temporary root, resolved lazily from the configuration."""
        if self._tmp_root is None:
            self._resolve_unset_roots()
        return cast("Path", self._tmp_root)

    @property
    def ledger(self) -> tuple[PathGrant, ...]:
        """Every path grant handed out so far, in grant order."""
        return self._ledger.grants

    def initialise(
        self,
        data_root: Path | None = None,
        cache_root: Path | None = None,
        tmp_root: Path | None = None,
    ) -> None:
        """
        Override roots; the single sanctioned injection point.

        Clears the usage ledger when any root actually changes, because
        outstanding grants refer to the previous roots. Calling without
        arguments keeps the currently active roots. Never creates
        directories in bulk — creation happens on access.
        """

        def changed_root(root: Path | None, current: Path | None) -> bool:
            """Whether the given root is a real change."""
            return root is not None and root != current

        changed = False

        if changed_root(data_root, self._data_root):
            self._data_root = data_root
            changed = True
        if changed_root(cache_root, self._cache_root):
            self._cache_root = cache_root
            changed = True
        if changed_root(tmp_root, self._tmp_root):
            self._tmp_root = tmp_root
            changed = True

        if changed:
            self._ledger.clear()
            self._root_map = None
            self._config_cache = None

    # --- Sliced getters ---
    def get_data_path(
        self,
        subpath: str = "",
        target_module: str | None = None,
        *,
        force: bool = False,
    ) -> Path:
        """Resolve a path under the data root for the calling module."""
        return self._resolve(PathRoot.DATA, subpath, target_module, force=force)

    def get_cache_path(
        self,
        subpath: str = "",
        target_module: str | None = None,
        *,
        force: bool = False,
    ) -> Path:
        """Resolve a path under the cache root for the calling module."""
        return self._resolve(PathRoot.CACHE, subpath, target_module, force=force)

    def get_tmp_path(
        self,
        subpath: str = "",
        target_module: str | None = None,
        *,
        force: bool = False,
    ) -> Path:
        """Resolve a path under the temporary root for the calling module."""
        return self._resolve(PathRoot.TMP, subpath, target_module, force=force)

    # --- Dynamic helpers ---
    def model_path(self, name: str, *, force: bool = False) -> Path:
        """Resolve a sanitised dynamic name under the shared models directory."""
        return self._dynamic(
            PathRoot.DATA,
            (self._config().model_subdirectory, "model_subdirectory"),
            (name, "model name"),
            force=force,
        )

    def raw_dataset_path(self, name: str, *, force: bool = False) -> Path:
        """Resolve a sanitised dynamic name under the shared raw datasets directory."""
        return self._dynamic(
            PathRoot.DATA,
            (self._config().raw_dataset_subdirectory, "raw_dataset_subdirectory"),
            (name, "raw dataset name"),
            force=force,
        )

    def processed_dataset_path(self, name: str, *, force: bool = False) -> Path:
        """Resolve a dynamic name under the shared processed datasets directory."""
        return self._dynamic(
            PathRoot.DATA,
            (
                self._config().preprocessed_dataset_subdirectory,
                "preprocessed_dataset_subdirectory",
            ),
            (name, "processed dataset name"),
            force=force,
        )

    def cache_path(self, name: str, *, force: bool = False) -> Path:
        """Resolve a sanitised dynamic name directly under the cache root."""
        return self._dynamic(
            PathRoot.CACHE,
            None,
            (name, "cache name"),
            force=force,
        )

    # NOTE: Shared-directory shortcuts (dynamic helpers) are defined here.
    # Docs: docs/UPS.md, "API quick reference" — add new shortcuts to that table too.

    def _dynamic(
        self,
        root_kind: PathRoot,
        subdirectory: tuple[str, str] | None,
        name: tuple[str, str],
        *,
        force: bool,
    ) -> Path:
        """
        Resolve a sanitised dynamic name under an optional shared subdirectory.

        `subdirectory` and `name` are (value, label) pairs; the labels name
        the arguments in validation errors. Validation order (subdirectory,
        then root, then name) matches the helpers this coalesces.

        Shared-dir exception: the helper's directory claim — and its
        ledger record — are attributed to the `shared` pseudo-slice, not to
        the caller's slice, because these locations are shared by design.
        """
        parts: tuple[str, ...] = ()
        if subdirectory is not None:
            subdirectory_value, subdirectory_label = subdirectory
            single_segment(subdirectory_value, label=subdirectory_label)
            parts = (subdirectory_value,)

        name_value, name_label = name
        resolved = self._root_for(root_kind).joinpath(
            *parts, self._sanitised(name_value, name_label)
        )
        return self._grant(root_kind, SHARED_SLICE, resolved, force=force)

    # --- Routing internals ---
    def _resolve(
        self,
        root_kind: PathRoot,
        subpath: str,
        target_module: str | None,
        *,
        force: bool,
    ) -> Path:
        """Validate, claim, materialise and record one sliced path grant."""
        if target_module is not None:
            module_slice = single_segment(target_module, label="target_module")
        else:
            module_slice = self._detect_caller_slice()

        segments = portable_segments(subpath, label="subpath")
        resolved = self._root_for(root_kind).joinpath(module_slice, *segments)
        return self._grant(root_kind, module_slice, resolved, force=force)

    def _grant(
        self,
        root_kind: PathRoot,
        module_slice: str,
        resolved: Path,
        *,
        force: bool,
    ) -> Path:
        """Claim, materialise and record one resolved path."""
        is_file = PurePosixPath(resolved.name).suffix != ""
        directory = resolved.parent if is_file else resolved

        self._ledger.claim(directory, module_slice, force=force)
        directory.mkdir(parents=True, exist_ok=True)
        self._ledger.record(
            PathGrant(
                root_kind=root_kind,
                module_slice=module_slice,
                resolved_path=resolved,
                forced=force,
            )
        )
        return resolved

    def _root_for(self, root_kind: PathRoot) -> Path:
        """Return the active root of the given kind (mapping built once)."""
        if self._root_map is None:
            self._root_map = {
                PathRoot.DATA: self.data_root,
                PathRoot.CACHE: self.cache_root,
                PathRoot.TMP: self.tmp_root,
            }
        return self._root_map[root_kind]

    def _detect_caller_slice(self) -> str:
        """Return the module slice of the first stack frame outside this package."""
        package = Path(__file__).resolve().parent
        frame = inspect.currentframe()
        while frame is not None:
            file_name = frame.f_globals.get("__file__")
            frame = frame.f_back

            if file_name is None:
                break
            resolved = Path(file_name).resolve()
            if resolved == package or package in resolved.parents:
                continue
            if not resolved.is_file():
                break
            try:
                canonical_id = self._environment.get_canonical_id(resolved)
            except ValueError:
                continue
            return self._slice_from_canonical_id(canonical_id)

        return INTERACTIVE_SLICE

    @staticmethod
    def _slice_from_canonical_id(canonical_id: str) -> str:
        """Return the slice encoded by a canonical id (first part after `src`)."""
        parts = [part for part in canonical_id.split("/") if part]
        if not parts:
            return INTERACTIVE_SLICE
        if len(parts) == 1:
            return PurePosixPath(parts[0]).stem
        return parts[0]

    @staticmethod
    def _sanitised(name: str, label: str) -> str:
        """Sanitise a dynamic name to one portable segment."""
        return single_segment(sanitise_segment(name), label)


# --- Singleton facade ---
"""Singleton router instance."""
paths = PathRouter()


def initialise(
    data_root: Path | None = None,
    cache_root: Path | None = None,
    tmp_root: Path | None = None,
) -> None:
    """Override the singleton router's roots; the only sanctioned injection point."""
    paths.initialise(data_root=data_root, cache_root=cache_root, tmp_root=tmp_root)


def get_data_path(
    subpath: str = "",
    target_module: str | None = None,
    *,
    force: bool = False,
) -> Path:
    """Resolve a data-root path through the singleton router."""
    return paths.get_data_path(subpath, target_module, force=force)


def get_cache_path(
    subpath: str = "",
    target_module: str | None = None,
    *,
    force: bool = False,
) -> Path:
    """Resolve a cache-root path through the singleton router."""
    return paths.get_cache_path(subpath, target_module, force=force)


def get_tmp_path(
    subpath: str = "",
    target_module: str | None = None,
    *,
    force: bool = False,
) -> Path:
    """Resolve a temporary-root path through the singleton router."""
    return paths.get_tmp_path(subpath, target_module, force=force)


def model_path(name: str, *, force: bool = False) -> Path:
    """Resolve a dynamic model name through the singleton router."""
    return paths.model_path(name, force=force)


def raw_dataset_path(name: str, *, force: bool = False) -> Path:
    """Resolve a dynamic raw dataset name through the singleton router."""
    return paths.raw_dataset_path(name, force=force)


def processed_dataset_path(name: str, *, force: bool = False) -> Path:
    """Resolve a dynamic processed dataset name through the singleton router."""
    return paths.processed_dataset_path(name, force=force)


def cache_path(name: str, *, force: bool = False) -> Path:
    """Resolve a dynamic cache name through the singleton router."""
    return paths.cache_path(name, force=force)

"""Central store: creates every module wishlist and serves resolved paths."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath

from src.argument_detection.paths import ArgumentDetectionPaths
from src.argument_framing.paths import ArgumentFramingPaths
from src.paths.core import (
    CorePaths,
    PathRoot,
    PathSpec,
    escapes_root,
)
from src.search_module.paths import SearchPaths

Wishlist = type[Enum]

ALL_WISHLISTS: tuple[Wishlist, ...] = (
    CorePaths,
    ArgumentDetectionPaths,
    ArgumentFramingPaths,
    SearchPaths,
)


@dataclass
class _Roots:
    """Holder for the active data/cache roots."""

    data: Path
    cache: Path


_roots: _Roots | None = None


def _configured_roots() -> _Roots:
    """
    Build the roots declared by `config.toml`.

    Imported lazily so that passing explicit roots never requires a config file.

    Returns:
        _Roots: The configured data and cache roots.
    """
    # Importing at module scope would read config.toml on every use,
    # including calls that pass both roots explicitly. It would
    # also create an import cycle, since src.configuration reaches these paths.
    from src.configuration import config  # pylint: disable=import-outside-toplevel

    return _Roots(data=config.data_directory, cache=config.cache_directory)


def _active_roots() -> _Roots:
    """
    Return the active roots. Defaults to the configured ones on first use.

    Returns:
        _Roots: The active roots.
    """
    # pylint: disable=global-statement  # Singleton by design.
    global _roots  # noqa: PLW0603
    if _roots is None:
        _roots = _configured_roots()
    return _roots


def initialise(
    data_root: Path | None = None,
    cache_root: Path | None = None,
) -> list[Path]:
    """
    Override the roots and create every path declared in module wishlists.

    Calling this without arguments keeps the currently active roots and only
    creates the missing directories, which makes it safe to call repeatedly.
    `config.toml` is read only when a root is not supplied, so passing both
    roots works from any working directory.

    The new roots become active only after every directory has been created, so
    a failed `mkdir` leaves the previous roots and the previous tree untouched.

    Args:
        data_root (Path | None): Override for the data root.
        cache_root (Path | None): Override for the cache root.

    Returns:
        list[Path]: All directories that exist after the call.
    """
    # pylint: disable=global-statement  # Singleton by design.
    global _roots  # noqa: PLW0603
    if data_root is None and cache_root is None:
        return ensure_all()
    if data_root is not None and cache_root is not None:
        prospective = _Roots(data=data_root, cache=cache_root)
    else:
        current = _roots if _roots is not None else _configured_roots()
        prospective = _Roots(
            data=data_root if data_root is not None else current.data,
            cache=cache_root if cache_root is not None else current.cache,
        )
    created = _ensure_all(prospective)
    _roots = prospective
    return created


def _resolve(member: Enum, roots: _Roots) -> Path:
    """
    Resolve a wishlist member against the given roots.

    Args:
        member (Enum): A member of any registered wishlist enum.
        roots (_Roots): Roots to resolve against.

    Returns:
        Path: Path under the data or cache root.

    Raises:
        ValueError: If `member` belongs to an unregistered wishlist.
        TypeError: If `member` does not hold a `PathSpec`.
    """
    if type(member) not in ALL_WISHLISTS:
        message = (
            f"{type(member).__name__} is not a registered wishlist; add it to "
            "ALL_WISHLISTS in src.paths.registry."
        )
        raise ValueError(message)

    spec = member.value
    if not isinstance(spec, PathSpec):
        message = f"Wishlist member {member!r} must hold a PathSpec."
        raise TypeError(message)

    root = roots.data if spec.root == PathRoot.DATA else roots.cache
    return root / spec.relative


def resolve(member: Enum) -> Path:
    """
    Resolve a wishlist member against the active root for its kind.

    Args:
        member (Enum): A member of any registered wishlist enum.

    Returns:
        Path: Path under the current data or cache root. Absolute only if the
            configured root is absolute.
    """
    return _resolve(member, _active_roots())


def _ensure_all(roots: _Roots) -> list[Path]:
    """
    Create directories for all wishlist entries against the given roots.

    File entries contribute their parent directory.

    Args:
        roots (_Roots): Roots to resolve against.

    Returns:
        list[Path]: Directories that exist after the call.
    """
    created: list[Path] = []
    for wishlist in ALL_WISHLISTS:
        for member in wishlist:
            spec: PathSpec = member.value  # type: ignore[attr-defined]
            path = _resolve(member, roots)
            target = path if spec.is_dir else path.parent
            target.mkdir(parents=True, exist_ok=True)
            if target not in created:
                created.append(target)

    return created


def ensure_all() -> list[Path]:
    """
    Create directories for all wishlist entries.

    File entries contribute their parent directory.

    Returns:
        list[Path]: Directories that exist after the call.
    """
    return _ensure_all(_active_roots())


def _is_single_segment(name: str) -> bool:
    """
    Report whether a dynamic name is exactly one path segment on every platform.

    Args:
        name (str): Dataset, model or cache name.

    Returns:
        bool: True if `name` is one segment under both POSIX and Windows rules.
    """
    return len(PurePosixPath(name).parts) == 1 and len(PureWindowsPath(name).parts) == 1


def _sanitise(name: str) -> str:
    """
    Normalise a dynamic dataset/model name to a single path segment.

    Args:
        name (str): Dataset, model or cache name.

    Returns:
        str: The name with spaces and hyphens replaced by underscores.

    Raises:
        ValueError: If `name` is empty, or would introduce a path separator, an
            absolute path, a `..` segment, or a trailing dot or space. Rejecting
            these keeps every resolved path inside its declared parent and names
            the same location on POSIX and Windows.
    """
    if (
        not name
        or not _is_single_segment(name)
        or escapes_root(name)
        or name.endswith((".", " "))
    ):
        message = f"Dynamic name must be a single path segment, got {name!r}."
        raise ValueError(message)

    return name.translate({ord(" "): "_", ord("-"): "_"})


def _dynamic(name: str, parent: Path) -> Path:
    """
    Resolve a sanitised dynamic name under a parent directory.

    Args:
        name (str): Dynamic dataset or model name.
        parent (Path): Resolved parent directory.

    Returns:
        Path: Path to the named entry inside `parent`.
    """
    return parent / _sanitise(name)


def raw_dataset_path(name: str) -> Path:
    """Resolve a dynamic raw-dataset directory under the shared root."""
    return _dynamic(name, resolve(CorePaths.RAW_DATASETS_DIR))


def processed_dataset_path(name: str) -> Path:
    """Resolve a dynamic processed-dataset directory under the shared root."""
    return _dynamic(name, resolve(CorePaths.PROCESSED_DATASETS_DIR))


def model_path(name: str) -> Path:
    """Resolve a model file under the shared models directory."""
    return _dynamic(name, resolve(CorePaths.MODELS_DIR))


def cache_path(name: str) -> Path:
    """Resolve a dynamic cache directory under the cache root."""
    return _dynamic(name, resolve(CorePaths.CACHES_DIR))

"""Paths owned by the project layout itself (shared by all modules)."""

from dataclasses import dataclass, replace
from enum import Enum, StrEnum, auto
from pathlib import PurePosixPath, PureWindowsPath
from typing import Self


class PathRoot(StrEnum):
    """Root a wishlist entry is resolved against."""

    DATA = auto()
    CACHE = auto()


def _segments(relative: str) -> tuple[str, ...]:
    """
    Return the segments a relative path names.

    relative = "" or "." yield no segmetns
    Callers will reject such declation with no location.

    Args:
        relative (str): Path relative to a root.

    Returns:
        tuple[str, ...]: The named segments.
    """
    return PurePosixPath(relative).parts


def escapes_root(relative: str) -> bool:
    """
    Report whether a relative path is absolute or steps above its root.

    Both POSIX and Windows rules are applied, so a path that stays inside the
    root on Linux also stays inside it on Windows.

    Args:
        relative (str): Path relative to a root.

    Returns:
        bool: True if the path is absolute or contains a `..` segment.
    """
    posix = PurePosixPath(relative)
    windows = PureWindowsPath(relative)
    return bool(
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or ".." in posix.parts
        or ".." in windows.parts
    )


def _reject(relative: str, label: str) -> None:
    """
    Raise ValueError unless a relative path is portable and is within its root.
    Portable between POSIX and Windows (backslash is a Windows separator).

    Args:
        relative (str): Path relative to a root.
        label (str): Name of the offending field, used in the error message.

    Raises:
        ValueError: If the path is empty, uses a backslash, ends with a dot or
            space, is absolute, or contains a `..` segment.
    """
    if not relative:
        message = f"{label} must not be empty."
        raise ValueError(message)

    if escapes_root(relative):
        message = f"{label} must stay inside its root, got {relative!r}."
        raise ValueError(message)
    if "\\" in relative:
        message = (
            f"{label} must use '/' as its separator, got {relative!r}; a backslash "
            "is a Windows separator and would nest there but not on POSIX."
        )
        raise ValueError(message)
    if relative.endswith((".", " ")):
        message = (
            f"{label} must not end with a dot or space, got {relative!r}; Windows "
            "strips them, so the name would differ between platforms."
        )
        raise ValueError(message)


@dataclass(frozen=True)
class PathSpec:
    """
    Relative declaration of a single required path.

    Supports `/` chaining like `pathlib.Path`: the child inherits
    - `root`
    - `is_dir`
    so mark file children with `.as_file()` if they are of file type.

    One whishlist cannot have two identical `(root, relative, is_dir)` tripples.
    If there is a duplicate then the duplicate becomes an alias of the first and
    is skipped by directory creation.

    Args:
        root (PathRoot): Root the path is resolved against.
        relative (str): Path relative to `root`, spelled with `/` separators so
            that it denotes the same location on POSIX and Windows. Never
            absolute.
        is_dir (bool): Whether the path denotes a directory rather than a file.

    Raises:
        ValueError: If `relative` escapes `root`, or is not spelled portably.
    """

    root: PathRoot
    relative: str
    is_dir: bool

    def __post_init__(self) -> None:
        """Validate that the declared relative path stays inside its root."""
        if self.relative == ".":
            # A directory may declare the root itself, as `CACHES_DIR` does.
            if not self.is_dir:
                message = "A file PathSpec.relative must name a location, got '.'."
                raise ValueError(message)
            return
        _reject(self.relative, "PathSpec.relative")
        if not self.is_dir and not _segments(self.relative):
            message = (
                f"A file PathSpec.relative must name a location, got {self.relative!r}."
            )
            raise ValueError(message)

    def __truediv__(self, sub: str) -> Self:
        """
        Chain a child segment onto this spec.

        Args:
            sub (str): Child segment to append.

        Returns:
            Self: New spec with the joined relative path.

        Raises:
            ValueError: If `sub` names no location, is not portable, is absolute,
                or traverses above the root. An empty or `.` child is rejected
                because `PurePosixPath` drops it, which would silently produce a
                second name for this same path.
        """
        if not _segments(sub):
            message = (
                f"Cannot chain an empty segment onto a PathSpec: {sub!r}; "
                "'' and '.' name no new location."
            )
            raise ValueError(message)
        _reject(sub, "A chained segment")
        joined = PurePosixPath(self.relative) / PurePosixPath(sub)
        return replace(self, relative=joined.as_posix())

    def as_file(self) -> Self:
        """Return a copy of this spec marked as a file."""
        return replace(self, is_dir=False)


class CorePaths(Enum):
    """Wishlist of shared data/cache directories."""

    MODELS_DIR = PathSpec(PathRoot.DATA, "models", is_dir=True)
    RAW_DATASETS_DIR = PathSpec(PathRoot.DATA, "raw_datasets", is_dir=True)
    PROCESSED_DATASETS_DIR = PathSpec(PathRoot.DATA, "processed_dataset", is_dir=True)
    # Written "." so dynamic cache directories sit directly under the cache root
    # rather than one level deeper.
    CACHES_DIR = PathSpec(PathRoot.CACHE, ".", is_dir=True)

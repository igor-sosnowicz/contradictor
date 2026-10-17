"""Safety guards shared by the unified path system."""

from enum import StrEnum, auto
from pathlib import PurePosixPath, PureWindowsPath


class PathRoot(StrEnum):
    """Root that resolved path is anchored at."""

    DATA = auto()
    CACHE = auto()
    TMP = auto()


def escapes_root(relative: str) -> bool:
    """
    Report whether a root-relative path is absolute or steps above its root.

    Both POSIX and Windows rules are applied, so a path that stays inside the
    root on Linux also stays inside it on Windows.

    Args:
        relative (str): Path relative to a root, spelled with '/' separators.

    Returns:
        bool: True if
            - path is absolute
            - carries a Windows drive
            - contains `..` segment.
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


def sanitise_segment(name: str) -> str:
    """
    Replace spaces and hyphens with underscores in a dynamic name.

    Args:
        name (str): Dynamic name (dataset or model name)

    Returns:
        str: The name with every space and hyphen replaced by an underscore.
    """
    return name.replace(" ", "_").replace("-", "_")


def portable_segments(relative: str, label: str) -> tuple[str, ...]:
    """
    Validate a root-relative path and return its POSIX segments.

    The empty string and `.` name the root directory itself and yield no
    segments. The path must be portable between POSIX and Windows: backslash
    is rejected because it is a Windows separator, and a trailing dot or space
    is rejected because Windows strips it.

    Args:
        relative (str): Path relative to a root, spelled with '/' separators.
        label (str): Name of the offending argument, used in error messages.

    Returns:
        tuple[str, ...]: The POSIX segments of the path; empty when it names
            the root itself.

    Raises:
        ValueError: If the path is absolute, contains a `..` segment, uses a
            backslash, has an empty or `.` segment, or ends with a dot or
            space.
    """
    if relative in ("", "."):
        return ()

    if "\\" in relative:
        message = (
            f"{label} must use '/' as its separator, got {relative!r}; a backslash "
            "is a Windows separator and would nest there but not on POSIX."
        )
        raise ValueError(message)

    if escapes_root(relative):
        message = f"{label} must stay inside its root, got {relative!r}."
        raise ValueError(message)

    # PurePosixPath collapses `a//b` and `a/./b` into `a/b`, which would
    # silently rename the location, so the raw spelling is checked too.
    if any(segment in ("", ".") for segment in relative.split("/")):
        message = (
            f"{label} must not contain empty or '.' segments, got {relative!r}; "
            "they would be dropped and silently rename the location."
        )
        raise ValueError(message)

    final = PurePosixPath(relative).parts[-1]
    if final.endswith((".", " ")):
        message = (
            f"{label} must not end with a dot or space, got {relative!r}; Windows "
            "strips them, so the name would differ between platforms."
        )
        raise ValueError(message)

    return PurePosixPath(relative).parts


def single_segment(segment: str, label: str) -> str:
    """
    Validate that a dynamic name is exactly one portable path segment.

    Args:
        segment (str): Expected to be a single segment (like "dataset name").
        label (str): Name of the offending argument, used in error messages.

    Returns:
        str: The validated segment.

    Raises:
        ValueError: If
            - name is empty
            - name is not portable
            - name spans multiple segments.
    """
    parts = portable_segments(segment, label)

    if len(parts) != 1:
        message = f"{label} must be a single path segment, got {segment!r}."
        raise ValueError(message)

    return parts[0]

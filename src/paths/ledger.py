"""
Usage ledger: granted paths and per-directory module-slice claims.

The collision rule lives here. Usage contract: `docs/UPS.md`.
"""

from dataclasses import dataclass
from pathlib import Path

from src.paths.core import PathRoot


@dataclass(frozen=True)
class PathGrant:
    """A single usage-ledger entry: one granted path."""

    root_kind: PathRoot
    module_slice: str
    resolved_path: Path
    forced: bool = False


class GrantLedger:
    """
    Owns the grants list and the per-directory slice claims.

    Collision rule: a directory claimed by two different module
    slices raises unless `force=True` allows the override. Full contract:
    `docs/UPS.md`.
    """

    def __init__(self) -> None:
        """Initialise an empty ledger."""
        self._grants: list[PathGrant] = []
        self._claims: dict[Path, str] = {}

    @property
    def grants(self) -> tuple[PathGrant, ...]:
        """Every recorded grant, in grant order."""
        return tuple(self._grants)

    def record(self, grant: PathGrant) -> None:
        """Append one granted path to the ledger."""
        self._grants.append(grant)

    def claim(self, directory: Path, module_slice: str, *, force: bool) -> None:
        """
        Record a module slice's claim on a physical directory.

        Args:
            directory (Path): The physical directory being claimed.
            module_slice (str): Module slice claiming the directory.
            force (bool): Whether an existing claim by a different slice may
                be overridden.

        Raises:
            ValueError: If a different module slice already claims the
                directory and `force` is False.
        """
        owner = self._claims.get(directory)
        if owner is None:
            self._claims[directory] = module_slice
            return
        if owner == module_slice or force:
            return

        message = (
            f"Directory {directory} is already claimed by module slice {owner!r}; "
            f"module slice {module_slice!r} would silently share it. "
            "Pass force=True to allow the override."
        )
        raise ValueError(message)

    def clear(self) -> None:
        """Drop all grants and claims (roots changed; grants are stale)."""
        self._grants.clear()
        self._claims.clear()

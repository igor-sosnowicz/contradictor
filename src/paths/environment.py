"""
Repository environment: root anchoring and canonical module identifiers.

Usage contract: `docs/UPS.md`.
"""

from pathlib import Path


class RepositoryEnvironment:
    """
    Locate the repository root and derive canonical module identifiers.

    Root anchoring scans upward from a caller's file path until a sentinel
    (`pyproject.toml` or `.git`) is found, and the first root found is
    cached. Full contract: `docs/UPS.md`.
    """

    _SENTINELS: tuple[str, ...] = ("pyproject.toml", ".git")

    def __init__(self) -> None:
        """Initialise an environment with an unresolved repository root."""
        self._repo_root: Path | None = None

    def repo_root_for(self, start: Path) -> Path:
        """Return the repository root containing the given file (cached)."""
        if self._repo_root is not None:
            return self._repo_root

        resolved = Path(start).resolve()
        for candidate in (resolved, *resolved.parents):
            if any((candidate / sentinel).exists() for sentinel in self._SENTINELS):
                self._repo_root = candidate
                return candidate

        sentinel_list = ", ".join(self._SENTINELS)
        message = f"No repository sentinel ({sentinel_list}) found above {resolved}."
        raise ValueError(message)

    def get_canonical_id(self, file_path: Path) -> str:
        """Return the file's repo-relative, `/`-joined id with `src` stripped."""
        resolved = Path(file_path).resolve()
        root = self.repo_root_for(resolved)

        try:
            relative = resolved.relative_to(root)
        except ValueError as e:
            message = f"{resolved} is not inside the repository rooted at {root}."
            raise ValueError(message) from e

        parts = relative.parts
        if parts and parts[0] == "src":
            parts = parts[1:]

        return "/".join(parts)

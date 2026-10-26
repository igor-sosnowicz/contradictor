"""Repository environment: root anchoring and canonical module identifiers."""

from pathlib import Path


class RepositoryEnvironment:
    """
    Locate the repository root and derive canonical module identifiers.

    Root anchoring scans upward from a caller's file path until a sentinel
    (`pyproject.toml` or `.git`) is found, and the first root found is
    cached.
    """

    _SENTINELS: tuple[str, ...] = ("pyproject.toml", ".git")

    def __init__(self) -> None:
        """Initialise an environment with an unresolved repository root."""
        self._repo_root: Path | None = None

    def _repo_root_for(self, start: Path) -> Path:
        """Return the repository root containing the given file."""
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
        """
        Return the file's repo-relative, `/`-joined id with `src` stripped.

        Args:
            file_path (Path): A file inside the repository.

        Returns:
            str: The canonical identifier, e.g. `search_module/cache`.

        Raises:
            ValueError: If the file lies outside the repository.
        """
        resolved = Path(file_path).resolve()
        root = self._repo_root_for(resolved)

        try:
            relative = resolved.relative_to(root)
        except ValueError as e:
            message = f"{resolved} is not inside the repository rooted at {root}."
            raise ValueError(message) from e

        parts = relative.parts
        if parts and parts[0] == "src":
            parts = parts[1:]

        return "/".join(parts)

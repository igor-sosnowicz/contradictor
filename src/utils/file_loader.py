r"""
Tool for loading files & conversion to plain text.

Supports ``.txt``, ``.md``/``.markdown``, ``.pdf`` and ``.html``/``.htm``
sources collected from a directory and/or an explicit file list. Every file
is normalised to plain ``.txt``-style text (``\\r\\n`` -> ``\\n``,
trailing whitespace stripped, 3+ blank lines collapsed, outer whitespace
stripped) so downstream code can treat all sources uniformly.

Performance notes:

- :meth:`SourceFileLoader.load` is a lazy generator: files are read and
  converted one by one, keeping peak memory at a single document.
- ``cache_dir`` stores the converted text next to nothing else, so a corpus of
  long PDFs is parsed once and read back as plain text afterwards. The key
  carries each file's size and mtime, so editing a source misses on its own.
- :meth:`SourceFileLoader.load_all` optionally uses a thread pool
  (I/O-bound work), giving near-linear speed-up for many files.
- Regexes are pre-compiled at module level; PDF page texts are joined
  (no quadratic ``+=`` in a loop).
"""

from __future__ import annotations

import hashlib
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import markdown as md_lib
from bs4 import BeautifulSoup, ResultSet, Tag
from loguru import logger
from pypdf import PdfReader
from pypdf.errors import PdfReadError

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator, Sequence

__all__: list[str] = ["SourceFile", "SourceFileLoader"]

_DEFAULT_EXTENSIONS: tuple[str, ...] = (
    ".txt",
    ".md",
    ".markdown",
    ".pdf",
    ".html",
    ".htm",
)

_BLANK_LINES_RE: re.Pattern[str] = re.compile(r"\n{3,}")
_TRAILING_WS_RE: re.Pattern[str] = re.compile(r"[ \t\f\v]+(?=\n)")


@dataclass(frozen=True)
class SourceFile:
    """A single loaded source document normalised to plain text."""

    path: Path
    name: str
    content: str


class SourceFileLoader: # pylint: disable=too-many-instance-attributes
    """
    Collect source paths and lazily load them as plain-text :class:`SourceFile`.

    Args:
        dir_path: Optional directory to scan for supported files.
        files: Optional explicit file paths (any supported suffix).
        recursive: If True, scan ``dir_path`` recursively (``rglob``).
            Otherwise only the top level is scanned (``glob``).
        extensions: File suffixes to accept (case-insensitive, with dot).
        encoding: Text encoding for ``.txt``/``.md``/``.html`` files.
        errors: Decoding error strategy (``"replace"`` never crashes).
        skip_empty: Skip files whose normalised text is empty (with warning).
        sort: Sort collected paths for deterministic order.
        cache_dir: Optional directory holding converted plain text. Extracting
            a few hundred pages of PDF costs seconds per file and the result
            never changes while the file does not, so a second run reads the
            cache instead. None disables caching.
    """

    def __init__(  # noqa: PLR0913, pylint: disable=too-many-arguments
        self,
        dir_path: Path | str | None = None,
        files: Sequence[Path | str] | None = None,
        *,
        recursive: bool = False,
        extensions: Sequence[str] = _DEFAULT_EXTENSIONS,
        encoding: str = "utf-8",
        errors: str = "replace",
        skip_empty: bool = True,
        sort: bool = True,
        cache_dir: Path | str | None = None,
    ) -> None:
        """Create a loader for the given source directory and/or file list."""
        self.dir_path: Path | None = Path(dir_path) if dir_path is not None else None
        self.files: list[Path] = [Path(f) for f in files] if files else []
        self.recursive: bool = recursive
        self.extensions: frozenset[str] = frozenset(
            e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions
        )
        self.encoding: str = encoding
        self.errors: str = errors
        self.skip_empty: bool = skip_empty
        self.sort: bool = sort
        self.cache_dir: Path | None = Path(cache_dir) if cache_dir is not None else None

        if self.dir_path is not None and not self.dir_path.is_dir():
            msg: str = f"Source directory does not exist: {self.dir_path}"
            raise FileNotFoundError(msg)

        self._parsers: dict[str, Callable[[Path], str]] = {
            ".pdf": self._pdf_to_text,
            ".md": self._md_to_text,
            ".markdown": self._md_to_text,
            ".html": self._html_to_text,
            ".htm": self._html_to_text,
            ".txt": self._read_text_file,
        }

        self.source_paths: list[Path] = self._collect_paths()

    def __len__(self) -> int:
        """Get the number of collected source files."""
        return len(self.source_paths)

    def __iter__(self) -> Iterator[SourceFile]:
        """Iterate over collected files."""
        yield from self.load()

    def load(self) -> Generator[SourceFile]:
        """Load the collected via Generator."""
        for path in self.source_paths:
            try:
                content: str = self._read_file(path)
            except (FileNotFoundError, ValueError, OSError, PdfReadError) as exc:
                logger.warning(f"Skipping source file {path}: {exc}")
                continue
            if self.skip_empty and not content:
                logger.warning(f"Skipping empty source file: {path.name}")
                continue
            yield SourceFile(path=path, name=path.name, content=content)

    def load_all(
        self, *, parallel: bool = False, max_workers: int | None = None
    ) -> list[SourceFile]:
        """
        Load all collected files into a list.

        Args:
            parallel: If True, read/convert files in a thread pool.
            max_workers: Thread pool size (None = auto).

        Returns:
            List of :class:`SourceFile` in collected order (parallel mode
            preserves input order).
        """
        if not parallel:
            return list(self.load())
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            ordered: list[SourceFile | None] = list(
                pool.map(self._load_one_or_none, self.source_paths)
            )
        return [doc for doc in ordered if doc is not None]

    # --- Path collection ---

    def _collect_paths(self) -> list[Path]:
        seen: dict[Path, None] = {}

        if self.dir_path is not None:
            patterns: list[str] = [f"*{ext}" for ext in self.extensions] + [
                f"*{ext.upper()}" for ext in self.extensions
            ]

            for pattern in dict.fromkeys(patterns):
                globber: Callable[..., Iterator[Path]] = (
                    self.dir_path.rglob if self.recursive else self.dir_path.glob
                )
                for path in globber(pattern):
                    if path.is_file() and path.suffix.lower() in self.extensions:
                        seen.setdefault(path.resolve(), None)

        for file in self.files:
            if file.suffix.lower() not in self.extensions:
                logger.warning(f"Skipping unsupported extension: {file}")
                continue
            if not file.is_file():
                logger.warning(f"Skipping missing source file: {file}")
                continue
            seen.setdefault(file.resolve(), None)

        paths: list[Path] = list(seen.keys())
        if self.sort:
            paths.sort(key=lambda p: p.name)
        return paths

    # --- Reading / dispatch ---

    def _load_one_or_none(self, path: Path) -> SourceFile | None:
        try:
            content: str = self._read_file(path)
        except (FileNotFoundError, ValueError, OSError, PdfReadError) as exc:
            logger.warning(f"Skipping source file {path}: {exc}")
            return None

        if self.skip_empty and not content:
            logger.warning(f"Skipping empty source file: {path.name}")
            return None
        return SourceFile(path=path, name=path.name, content=content)

    def _read_file(self, path: Path) -> str:
        if not path.is_file():
            msg: str = f"File not found: {path}"
            raise FileNotFoundError(msg)

        suffix: str = path.suffix.lower()
        if suffix not in self.extensions:
            msg = f"Unsupported file extension {suffix!r}: {path.name}"
            raise ValueError(msg)

        cached: Path | None = self._cache_path(path)
        if cached is not None and cached.is_file():
            return cached.read_text(encoding="utf-8", errors=self.errors)

        parser: Callable[[Path], str] = self._parsers.get(suffix, self._read_text_file)
        text: str = self._normalize(parser(path))
        if cached is not None:
            self._write_cache(cached, text)
        return text

    def _cache_path(self, path: Path) -> Path | None:
        """Locate the cache entry of one source file, or None when disabled."""
        if self.cache_dir is None:
            return None
        stat = path.stat()
        key: str = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
        digest: str = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{path.stem}-{digest}.txt"

    @staticmethod
    def _write_cache(cached: Path, text: str) -> None:
        """Write a cache entry, leaving no half-written file behind on failure."""
        cached.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path = cached.with_suffix(f".{os.getpid()}.tmp")
        try:
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(cached)
        except OSError as exc:
            logger.warning(f"Could not cache converted text for {cached.name}: {exc}")
            temporary.unlink(missing_ok=True)

    def _read_text_file(self, path: Path) -> str:
        return path.read_text(encoding=self.encoding, errors=self.errors)

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = _TRAILING_WS_RE.sub("", text)
        text = _BLANK_LINES_RE.sub("\n\n", text)
        return text.strip()

    # --- Private converters ---

    def _pdf_to_text(self, pdf_path: Path) -> str:
        reader: PdfReader = PdfReader(str(pdf_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    def _md_to_text(self, md_path: Path) -> str:
        raw: str = md_path.read_text(encoding=self.encoding, errors=self.errors)
        html: str = md_lib.markdown(raw)
        soup: BeautifulSoup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator="\n")

    def _html_to_text(self, html_path: Path) -> str:
        raw: str = html_path.read_text(encoding=self.encoding, errors=self.errors)
        soup: BeautifulSoup = BeautifulSoup(raw, "html.parser")
        unwanted: ResultSet[Tag] = soup(["script", "style", "noscript"])
        for tag in unwanted:
            tag.decompose()
        return soup.get_text(separator="\n")

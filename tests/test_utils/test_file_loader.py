"""Module for testing the file loader."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from src.utils.file_loader import SourceFile, SourceFileLoader

_FAKE_PDF: bytes = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200]
/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length 44 >> stream
BT /F1 12 Tf 10 10 Td (Hello PDF) Tj ET
endstream endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
trailer << /Root 1 0 R >>
startxref
0
%%EOF
"""

_FAKE_FILES: dict[str, str | bytes] = {
    "sample.txt": "Hello txt world.\nSecond line.\n",
    "sample.md": "# Title\n\nSome **bold** and [a link](https://example.com).\n",
    "sample.markdown": "Markdown ext works.\n",
    "sample.pdf": _FAKE_PDF,
    "sample.html": (
        "<html><head><style>p {color:red}</style></head><body>"
        "<h1>Hey</h1><script>evil()</script><p>Plain</p></body></html>"
    ),
    "sample.htm": "<html><body><p>Htm ext works.</p></body></html>",
}

_EXPECTED_CONTENT: dict[str, str] = {
    "sample.txt": "Hello txt world.\nSecond line.",
    "sample.md": "Title\n\nSome\nbold\n and\na link\n.",
    "sample.markdown": "Markdown ext works.",
    "sample.pdf": "Hello PDF",
    "sample.html": "Hey\nPlain",
    "sample.htm": "Htm ext works.",
}


@pytest.fixture
def temp_dir() -> Generator[Path]:
    """Fixture for an isolated temp directory, removed after the test."""
    with tempfile.TemporaryDirectory(prefix="contradictor-test-") as directory:
        yield Path(directory)


@pytest.fixture
def fake_sources_dir(temp_dir: Path) -> Path:
    """Fixture for a directory with one file per supported extension."""
    for name, content in _FAKE_FILES.items():
        path: Path = temp_dir / name
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
    return temp_dir


def test_loads_one_file_per_extension(fake_sources_dir: Path) -> None:
    """Test if every supported extension loads with the expected content."""
    loader: SourceFileLoader = SourceFileLoader(dir_path=fake_sources_dir)
    assert len(loader) == len(_EXPECTED_CONTENT)
    loaded: dict[str, str] = {doc.name: doc.content for doc in loader.load()}
    assert loaded == _EXPECTED_CONTENT


def test_txt_normalization(temp_dir: Path) -> None:
    """Test if plain text is normalised to the .txt plain format."""
    path: Path = temp_dir / "messy.txt"
    path.write_text("Hello  \r\n\r\n\r\n\r\nworld  \n", encoding="utf-8")
    loaded: list[SourceFile] = SourceFileLoader(files=[path]).load_all()
    assert len(loaded) == 1
    assert loaded[0].content == "Hello\n\nworld"


def test_skips_unsupported_and_missing_files(
    fake_sources_dir: Path, temp_dir: Path
) -> None:
    """Test if unsupported extensions and missing files are skipped."""
    skipped: Path = temp_dir / "notes.bin"
    skipped.write_text("binary", encoding="utf-8")
    loader: SourceFileLoader = SourceFileLoader(
        dir_path=fake_sources_dir,
        files=[skipped, fake_sources_dir / "no-such-file.txt"],
    )
    assert len(loader) == len(_EXPECTED_CONTENT)
    assert {doc.name for doc in loader.load()} == set(_EXPECTED_CONTENT)


def test_empty_file_is_skipped_by_default(temp_dir: Path) -> None:
    """Test if empty files are skipped unless skip_empty is disabled."""
    path: Path = temp_dir / "empty.txt"
    path.write_text("   \n  \n", encoding="utf-8")
    assert SourceFileLoader(files=[path]).load_all() == []
    kept: list[SourceFile] = SourceFileLoader(files=[path], skip_empty=False).load_all()
    assert len(kept) == 1
    assert kept[0].content == ""


def test_recursive_scan_finds_nested_files(fake_sources_dir: Path) -> None:
    """Test if recursive scan picks up files from subdirectories."""
    nested: Path = fake_sources_dir / "nested"
    nested.mkdir()
    (nested / "deep.txt").write_text("Deep content.\n", encoding="utf-8")
    assert len(SourceFileLoader(dir_path=fake_sources_dir)) == len(_EXPECTED_CONTENT)
    loader: SourceFileLoader = SourceFileLoader(
        dir_path=fake_sources_dir, recursive=True
    )
    assert len(loader) == len(_EXPECTED_CONTENT) + 1
    loaded: dict[str, str] = {doc.name: doc.content for doc in loader.load()}
    assert loaded["deep.txt"] == "Deep content."


def test_cache_dir_reuses_converted_text(temp_dir: Path) -> None:
    """Converted text is cached, and the source is not re-read on a second load."""
    source: Path = temp_dir / "report.txt"
    source.write_text("Hello world", encoding="utf-8")
    cache_dir: Path = temp_dir / "cache"

    first: list[SourceFile] = SourceFileLoader(
        files=[source], cache_dir=cache_dir
    ).load_all()
    assert first[0].content == "Hello world"
    cached: Path = next(iter(cache_dir.glob("report-*.txt")))

    cached.write_text("Planted", encoding="utf-8")
    loaded: list[SourceFile] = SourceFileLoader(
        files=[source], cache_dir=cache_dir
    ).load_all()
    assert loaded[0].content == "Planted"


def test_cache_key_follows_the_source_file(temp_dir: Path) -> None:
    """Editing a source moves its cache key, so a stale entry is never served."""
    source: Path = temp_dir / "report.txt"
    source.write_text("First version", encoding="utf-8")
    cache_dir: Path = temp_dir / "cache"
    SourceFileLoader(files=[source], cache_dir=cache_dir).load_all()

    source.write_text("Second version, a different size entirely", encoding="utf-8")
    loaded: list[SourceFile] = SourceFileLoader(
        files=[source], cache_dir=cache_dir
    ).load_all()
    assert loaded[0].content == "Second version, a different size entirely"
    assert len(list(cache_dir.glob("report-*.txt"))) == 2


def test_no_cache_dir_writes_nothing(temp_dir: Path) -> None:
    """Caching is opt-in: without cache_dir nothing is written to disk."""
    source: Path = temp_dir / "report.txt"
    source.write_text("Hello world", encoding="utf-8")
    SourceFileLoader(files=[source]).load_all()
    assert sorted(p.name for p in temp_dir.iterdir()) == ["report.txt"]

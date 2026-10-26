"""Script for formatting examples in a consistent way."""

import argparse
import re
from pathlib import Path


def normalise(text: str) -> str:
    """
    Convert a text into a consistent format.

    Args:
        text (str): A text with formatting issues.

    Returns:
        str: A text without formatting issues.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Treat lines indented by four or more spaces as new paragraphs.
    text = re.sub(r"\n[ \t]{4,}(?=\S)", "\n\n", text)

    # Remove other line-edge whitespace and normalise paragraph breaks.
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Join remaining single newlines, which are treated as line wraps.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)

    # Remove bracketed numeric citations, including lists like [1, 2].
    text = re.sub(r"\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\]", "", text)

    # Remove bare citation numbers after sentence punctuation, e.g. "claim.4 Next".
    # The decimal guard avoids deleting the fractional part of values like "3.14".
    text = re.sub(r"(?<!\d[.!?])(?<=[.!?])\d{1,3}(?=\s|$)", "", text)

    # Remove bare citation numbers before punctuation, e.g. "claim4."
    text = re.sub(r"(?<=[A-Za-z])\d{1,3}(?=[.,!?;:])", "", text)

    return text.strip() + "\n"


def normalise_file(path: Path) -> None:
    """
    Normalise content of a UTF-8-encoded text file.

    Args:
        path (Path): Path to a file.
    """
    text = path.read_text(encoding="utf-8")
    path.write_text(normalise(text), encoding="utf-8")
    print(f"Normalised: {path}")  # noqa: T201, Acceptable in a script.


def main() -> None:
    """Parse arguments, normalise files to a consistent format."""
    parser = argparse.ArgumentParser(
        description="Normalise line breaks and whitespace in text files."
    )
    parser.add_argument("path", type=Path, help="Path to a text file or directory")
    args = parser.parse_args()

    if args.path.is_file():
        normalise_file(args.path)
    elif args.path.is_dir():
        files = sorted(args.path.glob("*.txt"))
        if not files:
            print(f"No .txt files found in: {args.path}")  # noqa: T201, Allowed in scripts.
            return
        for path in files:
            normalise_file(path)
    else:
        parser.error(f"File or directory not found: {args.path}")


if __name__ == "__main__":
    main()

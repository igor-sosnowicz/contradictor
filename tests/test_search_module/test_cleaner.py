"""Tests for cleaner in search module."""

from src.search_module.cleaner import BeautifulSoupCleaner
from src.search_module.config import CleaningConfig

HTML = """
<html>
<head>
<title>Example</title>
</head>

<body>

<nav>
MENU
</nav>

<script>
alert(1)
</script>

<article>
Cats are wonderful pets.
Cats like sleeping.
Cats are independent.
Cats enjoy playing.
Cats are popular animals.
Cats live with humans.
Cats are friendly.
Cats are intelligent.
Cats can hunt mice.
Cats purr.
</article>

</body>
</html>
"""


def test_empty_html() -> None:
    """Verify that empty HTML produces an empty document."""
    cleaner = BeautifulSoupCleaner(
        CleaningConfig(
            min_text_length=0,
            min_word_count=0,
            min_sentence_count=0,
        )
    )

    document = cleaner.clean("", "https://example.com")

    assert document.text == ""


def test_extract_title() -> None:
    """Verify that page title is extracted from HTML content."""
    cleaner = BeautifulSoupCleaner(
        CleaningConfig(
            min_text_length=0,
            min_word_count=0,
            min_sentence_count=0,
        )
    )

    document = cleaner.clean(
        HTML,
        "https://example.com",
    )

    assert document.title == "Example"


def test_remove_navigation() -> None:
    """Verify that navigation elements are removed from cleaned content."""
    cleaner = BeautifulSoupCleaner(
        CleaningConfig(
            min_text_length=0,
            min_word_count=0,
            min_sentence_count=0,
        )
    )

    document = cleaner.clean(
        HTML,
        "https://example.com",
    )

    assert "MENU" not in document.text
    assert "Cats are wonderful pets." in document.text

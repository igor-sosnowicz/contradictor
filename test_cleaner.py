"""Ad-hoc script to verify HTML cleaning functionality on a sample page structure."""

from loguru import logger

from src.search_module.cleaner import ReadabilityCleaner
from src.search_module.config import CleaningConfig

html = """
<html>
<header>
Menu
</header>

<body>

<script>
alert("test")
</script>

<nav>
Home About
</nav>

<article>
Cats are better pets than dogs.
</article>

<footer>
Copyright
</footer>

</body>
</html>
"""


cleaner = ReadabilityCleaner(CleaningConfig())

doc = cleaner.clean(
    html,
    "https://example.com",
)

logger.info(doc.text)

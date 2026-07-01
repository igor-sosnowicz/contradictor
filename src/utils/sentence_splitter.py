"""Module with sentence splitting utility."""

from collections.abc import Iterable

from spacy.lang.en import English


class SentenceSplitter:  # pylint: disable=too-few-public-methods
    """Split text into sentences."""

    def __init__(self) -> None:
        """Initialise a SpaCy pipeline for sentence splitting."""
        self._nlp = English()
        self._nlp.add_pipe("sentencizer")

    def __call__(self, text: str) -> Iterable[str]:
        """
        Split a text into sentences.

        Args:
            text (str): Text to be split into sentences.

        Returns:
            Iterable[str]: Sentences.
        """
        doc = self._nlp(text)
        return (sent.text for sent in doc.sents)

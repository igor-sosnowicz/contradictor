"""Module with an interface of an abstract argument detector."""

from abc import ABC, abstractmethod

from src.data_models.data_models import TextSpan


class ArgumentDetector(ABC):
    """An interface of an abstract argument detector."""

    @abstractmethod
    def detect(self, text: str) -> list[TextSpan]:
        """
        Detect arguments from a given text.

        Args:
            text (str): Text containing both argument and non-arguments.

        Returns:
            list[TextSpan]: Spans of the original text containing arguments.
        """

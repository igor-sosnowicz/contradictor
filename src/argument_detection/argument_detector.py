"""Module with an interface of an abstract argument detector."""

from abc import ABC, abstractmethod


class ArgumentDetector(ABC):
    """An interface of an abstract argument detector."""

    @abstractmethod
    async def detect(self, text: str) -> list[str]:
        """
        Detect arguments from a given text.

        Args:
            text (str): Text containing both argument and non-arguments.

        Returns:
            list[str]: Sentences of the original text containing arguments.
        """

    @abstractmethod
    async def _train(self) -> object: ...

    @abstractmethod
    def _load(self) -> object: ...

    @abstractmethod
    async def perform_tuning(self) -> dict[str, float]:
        """
        Perform hyperparameter tuning.

        Returns:
            dict[str, float]: Split name and metric name with corresponding
                metric value.
        """

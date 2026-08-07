"""Module with an interface of an abstract argument framer."""

from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable

from src.data_models.data_models import Argument, FramedArgument


class ArgumentFramer(ABC):
    """An interface of an abstract argument framer."""

    @abstractmethod
    def frame(self, arguments: Iterable[Argument]) -> Collection[FramedArgument]:
        """
        Assign the interpretative frame for a given argument.

        Args:
            arguments (Iterable[Argument]): An iterable with arguments to be framed.

        Returns:
            Collection[FramedArgument]: Arguments with predicted frames.
        """

    @abstractmethod
    async def prepare(self) -> None:
        """Prepare the model and dataset to perform inference."""

    @abstractmethod
    async def _train(self) -> object: ...

    @abstractmethod
    def _load(self) -> object: ...

    @abstractmethod
    async def perform_tuning(self) -> dict[str, float]:
        """
        Perform hyperparameter tuning for the framing classifier.

        Returns:
            dict[str, float]: Split name and metric name with corresponding
                metric value.
        """

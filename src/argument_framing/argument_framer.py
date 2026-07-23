"""Module with an interface of an abstract argument framer."""

from abc import ABC, abstractmethod

from src.argument_framing.argument_framing_dataset import ArgumentFramingDataset
from src.data_models.data_models import FramedArgument


class ArgumentFramer(ABC):
    """An interface of an abstract argument framer."""

    @abstractmethod
    async def frame(self, text: str) -> FramedArgument:
        """
        Assign the interpretative frame for a given argument.

        Args:
            text (str): Text of an argument.

        Returns:
            FramedArgument: Argument data model with predicted frame.
        """

    @abstractmethod
    async def _train(self, dataset: ArgumentFramingDataset) -> object: ...

    @abstractmethod
    def _load(self) -> object: ...

    @abstractmethod
    async def perform_tuning(self, dataset: ArgumentFramingDataset) -> dict[str, float]:
        """
        Perform hyperparameter tuning for the framing classifier.

        Args:
            dataset (ArgumentFramingDataset): Dataset containing
            training and validation data.

        Returns:
            dict[str, float]: Split name and metric name with corresponding
                metric value.
        """

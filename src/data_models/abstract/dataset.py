"""Module with an interface for a dataset."""

from abc import ABC, abstractmethod
from pathlib import Path


class Dataset(ABC):
    """An interface for extracting, transforming, and loading a custom dataset."""

    @abstractmethod
    async def prepare(self) -> Path:
        """
        Prepare a dataset by extracting raw datasets and transforming it into
        a usable dataset.

        Returns:
            Path: Path to a file or a directory containing the processed data.

        """

    @abstractmethod
    def clear(self) -> None:
        """Remove both raw and the processed data."""

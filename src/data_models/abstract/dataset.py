"""Module with an interface for a dataset."""

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from src.data_models.data_models import SubsetName


class Dataset(ABC):
    """An interface for extracting, transforming, and loading a custom dataset."""

    @abstractmethod
    def get_split(self, split: SubsetName) -> pd.DataFrame:
        """
        Get data for a given split.

        The dataset has to be prepared before getting a given split.

        Args:
            split (SubsetName): Split for which data should be provided.

        Returns:
            pd.DataFrame: Data frame with examples.
        """

    @abstractmethod
    async def prepare(self) -> Path:
        """
        Prepare a dataset by extracting raw datasets and transforming it into
        a usable dataset.
        """

    @abstractmethod
    def clear(self) -> None:
        """Remove both raw and the processed data."""

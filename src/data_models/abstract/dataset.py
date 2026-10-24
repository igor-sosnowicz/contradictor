"""Module with an interface for a dataset."""

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from src.data_models.data_models import SubsetName


class Dataset(ABC):
    """An interface for extracting, transforming, and loading a custom dataset."""

    def __init__(self, random_seed: int | None = None) -> None:
        """
        Initialise an empty dataset.

        Args:
            random_seed (int | None): A seed for a random number generator.
                Useful for reproducibility of results.
        """
        self._datasets: dict[Path, list[str]] = {}
        self._splits_path: Path | None = None
        self.random_seed = random_seed

    def get_split(self, split: SubsetName, max_samples: int = 0) -> pd.DataFrame:
        """
        Get a prepared dataset split.

        Args:
            split (SubsetName): Name of the dataset split to retrieve.
            max_samples (int, optional): Maximum number of samples to return.
                If set to 0, all samples are returned. Defaults to 0.

        Returns:
            pd.DataFrame: DataFrame containing the requested dataset split.

        Raises:
            DatasetError: If the dataset has not been prepared yet.
        """
        if isinstance(split, str):
            split = SubsetName(split)
        if self._splits_path is None:
            raise ValueError(
                "You have to prepare a dataset before getting one of its splits."
            )
        subset_file = self._splits_path / f"{split.value}.csv"
        df = pd.read_csv(subset_file)
        if max_samples != 0:
            df = df.sample(
                n=min(max_samples, len(df)),
                random_state=self.random_seed,
            )
        return df

    @property
    @abstractmethod
    def processed_folder_name(self) -> str:
        """
        Get the name of the directory for processed dataset splits.

        Returns:
            str: The name of a directory for processed dataset splits.
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

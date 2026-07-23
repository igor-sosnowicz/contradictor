"""Abstract dataset implementation for datasets downloaded from URLs."""

import asyncio
import shutil
from abc import abstractmethod
from pathlib import Path

import httpx
import pandas as pd
from loguru import logger

from src.configuration import config
from src.data_models.abstract.dataset import Dataset
from src.data_models.data_models import SubsetName


class UrlDataset(Dataset):
    """Base class for datasets downloaded from one or more URLs."""

    @property
    @abstractmethod
    def processed_folder_name(self) -> str:
        """Name of the directory where preprocessed splits are stored."""

    def __init__(self) -> None:
        """
        Initialise an empty URL-based dataset.

        Creates storage for dataset URLs and resets the path to processed
        dataset splits.
        """
        self._datasets: dict[Path, list[str]] = {}
        self._splits_path: Path | None = None

    async def prepare(self) -> Path:
        """
        Prepare the dataset by downloading, transforming, and saving splits.

        This method downloads raw datasets if they are missing, transforms them,
        and creates the processed dataset splits. If the dataset is already
        prepared, it sets the path to the existing processed dataset.

        Raises:
            DatasetError: If dataset preparation fails.
        """
        await self._download_raw_datasets_if_missing()
        datasets = self._transform()
        if not datasets:
            self._splits_path = (
                config.data_directory
                / config.preprocessed_dataset_subdirectory
                / self.processed_folder_name
            )
            return self._splits_path
        self._splits_path = self._merge_datasets(datasets)
        return self._splits_path

    def clear(self) -> None:
        """Remove downloaded raw datasets."""
        for dataset in self._datasets:
            if dataset.exists():
                shutil.rmtree(dataset)

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
        if self._splits_path is None:
            raise ValueError(
                "You have to prepare a dataset before getting one of its splits."
            )

        subset_file = self._splits_path / f"{split.value}.csv"
        df = pd.read_csv(subset_file)
        if max_samples != 0:
            df = df.sample(
                n=min(max_samples, len(df)),
                random_state=42,
            )
        return df

    async def _download_raw_datasets_if_missing(self) -> None:
        """Download raw datasets if missing."""
        logger.debug("Starting raw datasets download check")
        tasks = []

        async with httpx.AsyncClient() as client:
            for local_path, urls in self._datasets.items():
                local_path.mkdir(parents=True, exist_ok=True)

                for url in urls:
                    file_name = url.split("/")[-1]
                    file_path = local_path / file_name

                    if file_path.exists() and file_path.stat().st_size > 0:
                        logger.debug(f"Using cached version: {file_path}")
                        continue

                    logger.debug(f"Downloading: {url}")
                    tasks.append(
                        (
                            file_path,
                            client.get(
                                url=url,
                                timeout=config.dataset_download_timeout,
                                follow_redirects=True,
                            ),
                        )
                    )

            results = await asyncio.gather(*(request for _, request in tasks))

        for (file_path, _), response in zip(tasks, results, strict=True):
            response.raise_for_status()
            logger.debug(f"Downloaded {len(response.content)} bytes to: {file_path}")
            with file_path.open("wb") as file:
                file.write(response.content)

    def _merge_datasets(
        self, extracted_data: dict[Path, dict[SubsetName, pd.DataFrame]]
    ) -> Path:
        """Merge and save processed dataset splits."""
        merged_data = self._combine_subsets(extracted_data)

        processed_dataset_path = (
            config.data_directory
            / config.preprocessed_dataset_subdirectory
            / self.processed_folder_name
        )

        processed_dataset_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        for subset_name, df in merged_data.items():
            subset_file = processed_dataset_path / f"{subset_name.value}.csv"
            df.to_csv(subset_file, index=False)

        self._log_statistics(merged_data)

        return processed_dataset_path

    def _combine_subsets(
        self, extracted_data: dict[Path, dict[SubsetName, pd.DataFrame]]
    ) -> dict[SubsetName, pd.DataFrame]:
        merged_subsets: dict[SubsetName, list[pd.DataFrame]] = {
            SubsetName.TRAINING: [],
            SubsetName.VALIDATION: [],
            SubsetName.TESTING: [],
        }

        for subsets in extracted_data.values():
            for subset_name, df in subsets.items():
                if len(df) > 0:
                    merged_subsets[subset_name].append(df)

        merged_data = {}

        for subset_name, dfs in merged_subsets.items():
            if dfs:
                merged_df = pd.concat(dfs, ignore_index=True)

                merged_df = merged_df.sample(frac=1, random_state=42).reset_index(
                    drop=True
                )
                merged_data[subset_name] = merged_df
            else:
                merged_data[subset_name] = pd.DataFrame()

        return merged_data

    def _log_statistics(self, merged_data: dict[SubsetName, pd.DataFrame]) -> None:
        """Log dataset statistics."""
        total_records = sum(len(df) for df in merged_data.values())

        for subset_name, df in merged_data.items():
            logger.debug(f"{subset_name.value}: {len(df)} records")

        logger.debug(f"Total: {total_records} records")

    @abstractmethod
    def _transform(
        self,
    ) -> dict[Path, dict[SubsetName, pd.DataFrame]]:
        """Convert raw files into split DataFrames."""

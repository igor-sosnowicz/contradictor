"""Module with an argument framing dataset using direct URLs."""

from pathlib import Path
from typing import Final, override

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from src.data_models.abstract.url_dataset import UrlDataset
from src.data_models.data_models import SubsetName
from src.utils.dataset import to_raw_dataset_path


class ArgumentFramingDataset(UrlDataset):
    """Dataset for argument framing using direct Hugging Face data downloads."""

    MIN_CLASS_SAMPLES: Final = 2

    @property
    @override
    def processed_folder_name(self) -> str:
        """Get the name of directory for processed dataset."""
        return "argument_framing"

    def __init__(self, random_seed: int | None = None) -> None:
        """Define raw databases to be downloaded via direct URLs."""
        super().__init__()
        self.random_seed = random_seed
        self._datasets = {
            to_raw_dataset_path("mp_frame_corpus"): [
                "https://huggingface.co/datasets/jmLuis/"
                "MediaFrameCorpus-PhilippineFrameCorpus-Combined/"
                "resolve/main/combined_train.csv",
                "https://huggingface.co/datasets/jmLuis/"
                "MediaFrameCorpus-PhilippineFrameCorpus-Combined/"
                "resolve/main/combined_validation.csv",
            ],
        }
        self._splits_path = None

    @override
    def _transform(self) -> dict[Path, dict[SubsetName, pd.DataFrame]]:
        """Transform raw csv datasets into standardized split formats."""
        transformed_data = {}
        dataset_path = to_raw_dataset_path("argument-framing")

        if dataset_path.exists():
            transformed_data[dataset_path] = self._load_and_split_mfc()

        return transformed_data

    @override
    def _log_statistics(
        self,
        merged_data: dict[SubsetName, pd.DataFrame],
    ) -> None:
        """Log framing-specific statistics."""
        total_records = 0

        for subset_name, df in merged_data.items():
            total_records += len(df)

            logger.debug(f"{subset_name.value}: {len(df)} framed arguments")

            if len(df) > 0:
                top_frames = df["interpretative_frame"].value_counts().head(3)

                logger.debug(f"Top frames:\n{top_frames}")

        logger.debug(f"Total: {total_records} framed arguments")

    def _load_and_split_mfc(self) -> dict[SubsetName, pd.DataFrame]:
        """Load CSV files and split the validation file into val and test sets."""
        dataset_path = to_raw_dataset_path("argument-framing")
        train_file = dataset_path / "combined_train.csv"
        val_file = dataset_path / "combined_validation.csv"

        if not train_file.exists() or not val_file.exists():
            logger.error(f"Required raw files missing in {dataset_path}")
            return {}

        df_train = (
            pd.read_csv(train_file)
            .dropna(subset=["text", "label"])
            .rename(columns={"label": "interpretative_frame"})
        )

        df_val_raw = (
            pd.read_csv(val_file)
            .dropna(subset=["text", "label"])
            .rename(columns={"label": "interpretative_frame"})
        )

        class_counts = df_val_raw["interpretative_frame"].value_counts()
        valid_classes = class_counts[class_counts >= self.MIN_CLASS_SAMPLES].index
        df_val_raw = df_val_raw[df_val_raw["interpretative_frame"].isin(valid_classes)]

        df_val, df_test = train_test_split(
            df_val_raw,
            test_size=0.5,
            random_state=self.random_seed,
            stratify=df_val_raw["interpretative_frame"],
        )

        return {
            SubsetName.TRAINING: df_train[["text", "interpretative_frame"]],
            SubsetName.VALIDATION: df_val[["text", "interpretative_frame"]],
            SubsetName.TESTING: df_test[["text", "interpretative_frame"]],
        }

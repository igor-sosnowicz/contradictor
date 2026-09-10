"""Module with an argument detection dataset."""

import zipfile
from pathlib import Path
from typing import override

import pandas as pd
from loguru import logger
from sklearn.model_selection import GroupShuffleSplit

from src.argument_detection.config import XGBoostExtractorConfig
from src.configuration import config
from src.data_models.abstract.url_dataset import UrlDataset
from src.data_models.data_models import SubsetName
from src.utils.dataset import filter_empty_rows, to_raw_dataset_path
from src.utils.errors import DatasetError

MIN_CLAIM_WORDS = 4


class ArgumentDetectionDataset(UrlDataset):
    """
    Dataset for argument detection tasks.

    This dataset handles multiple argument-related corpora and prepares
    processed datasets for:
    - claim extraction,
    - evidence extraction,
    - argument detection.

    The class downloads raw datasets, cleans and merges subsets, creates
    derived datasets, performs leakage-free splitting, and provides access
    to processed dataset splits.
    """

    NUM_TRAIN_VAL_TEST_SPLITS = 1
    SAMPLE_ALL_RECORDS = 1.0

    @property
    @override
    def processed_folder_name(self) -> str:
        return "argument_detection"

    @staticmethod
    def _clean_sentence(sentence: str) -> str:
        """
        Clean a sentence by normalizing whitespace.

        Args:
            sentence (str): Input sentence to clean.

        Returns:
            str: Cleaned sentence with normalized whitespace.
        """
        cleaned = sentence.replace("\n", " ").replace("\r", " ")
        return " ".join(cleaned.split())

    def __init__(self) -> None:
        """Initialise the argument detection dataset."""
        # Persuade 1.0:
        # https://www.kaggle.com/datasets/julesking/tla-lab-persuade-dataset

        super().__init__()

        self._datasets = {
            to_raw_dataset_path("persuade"): [
                (
                    "https://www.kaggle.com/api/v1/datasets/download/"
                    "julesking/tla-lab-persuade-dataset"
                ),
            ],
        }

    @override
    def _combine_subsets(
        self,
        extracted_data: dict[Path, dict[SubsetName, pd.DataFrame]],
    ) -> dict[SubsetName, pd.DataFrame]:
        merged = super()._combine_subsets(extracted_data)

        for subset_name, dataframe in merged.items():
            if dataframe.empty:
                continue
            if "sentence" in dataframe.columns:
                cleaned_df = filter_empty_rows(dataframe, "sentence")
            elif "claim" in dataframe.columns and "evidence" in dataframe.columns:
                cleaned_df = filter_empty_rows(dataframe, ["claim", "evidence"])
            else:
                cleaned_df = dataframe
            merged[subset_name] = cleaned_df.reset_index(drop=True)
        return merged

    @override
    def _log_statistics(
        self,
        merged_data: dict[SubsetName, pd.DataFrame],
    ) -> None:
        for subset_name, df in merged_data.items():
            if df.empty:
                logger.warning(f"{subset_name}: empty dataset")
                continue

            logger.debug(f"Subset: {subset_name}")
            logger.debug(f"Rows: {len(df)}")

            if "is_evidence" in df.columns:
                positives = df["is_evidence"].sum()
                negatives = len(df) - positives

                logger.debug(
                    f"Evidence pairs: {positives}, Non-evidence pairs: {negatives}",
                )

            elif "is_argument" in df.columns:
                arguments = df["is_argument"].sum()
                non_arguments = len(df) - arguments

                logger.debug(
                    f"Arguments: {arguments}, Non-arguments: {non_arguments}",
                )

    def _load_persuade(self) -> dict[str, dict[SubsetName, pd.DataFrame]]:
        """
        Load the PERSUADE 2.0 corpus and create extraction datasets.

        Processes source essays and annotations to build separate training,
        validation, and testing splits for claim and evidence extraction.

        Returns:
            dict[str, dict[SubsetName, pd.DataFrame]]: Datasets grouped by
            extraction task and subset name.

        Raises:
            DatasetError: If the PERSUADE CSV file does not exist.
        """
        dataset_directory = to_raw_dataset_path("persuade")
        archive_path = dataset_directory / "tla-lab-persuade-dataset"

        if archive_path.exists() and zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as zip_ref:
                zip_ref.extractall(dataset_directory)

        csv_file = dataset_directory / "persuade2_train_srctexts.csv"

        if not csv_file.exists():
            raise DatasetError("Persuade CSV file was not found.")

        raw_df = pd.read_csv(csv_file, low_memory=False)

        claim_df = self._create_claim_dataset(raw_df)
        claim_splits = self._split_dataset(
            claim_df,
            group_column="essay_id_comp",
        )

        evidence_df = self._create_claim_evidence_pairs(raw_df)
        evidence_df = self._add_negative_pairs(evidence_df)

        evidence_splits = self._split_dataset(
            evidence_df,
            group_column="essay_id_comp",
        )
        return {
            "claim_extraction": {
                SubsetName.TRAINING: claim_splits[0],
                SubsetName.VALIDATION: claim_splits[1],
                SubsetName.TESTING: claim_splits[2],
            },
            "evidence_extraction": {
                SubsetName.TRAINING: evidence_splits[0],
                SubsetName.VALIDATION: evidence_splits[1],
                SubsetName.TESTING: evidence_splits[2],
            },
        }

    def _split_dataset(
        self,
        df: pd.DataFrame,
        group_column: str = "essay_id_comp",
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the dataset by essay to prevent data leakage.

        All samples from the same essay are kept in the same subset.

        Args:
            df (pd.DataFrame): Dataset to split.
            group_column (str): Column containing group identifiers used to
                prevent data leakage between subsets.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Training,
            validation, and testing dataframes.

        Raises:
            DatasetError: If the specified group column does not exist.
        """
        if df.empty:
            return df.copy(), df.copy(), df.copy()

        if group_column not in df.columns:
            raise DatasetError(
                "Persuade dataset requires 'essay_id_comp' for leakage-free splitting."
            )

        splitter = GroupShuffleSplit(
            n_splits=self.NUM_TRAIN_VAL_TEST_SPLITS,
            test_size=XGBoostExtractorConfig.dataset.test_size,
            random_state=XGBoostExtractorConfig.dataset.random_state,
        )

        train_idx, temp_idx = next(
            splitter.split(df, groups=df[group_column]),
        )

        train_df = df.iloc[train_idx].copy()
        temp_df = df.iloc[temp_idx].copy()
        splitter_temp = GroupShuffleSplit(
            n_splits=self.NUM_TRAIN_VAL_TEST_SPLITS,
            test_size=XGBoostExtractorConfig.dataset.val_size,
            random_state=XGBoostExtractorConfig.dataset.random_state,
        )

        val_idx, test_idx = next(
            splitter_temp.split(
                temp_df,
                groups=temp_df[group_column],
            ),
        )

        val_df = temp_df.iloc[val_idx].copy()
        test_df = temp_df.iloc[test_idx].copy()

        train_df = train_df.drop(
            columns=["essay_id_comp"],
            errors="ignore",
        )

        val_df = val_df.drop(
            columns=["essay_id_comp"],
            errors="ignore",
        )

        test_df = test_df.drop(
            columns=["essay_id_comp"],
            errors="ignore",
        )
        return (
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    def _transform(self) -> dict:
        """Create two processed datasets for claim-evidence extraction."""
        claim_path = (
            config.data_directory
            / config.preprocessed_dataset_subdirectory
            / "claim_extraction"
        )
        evidence_path = (
            config.data_directory
            / config.preprocessed_dataset_subdirectory
            / "evidence_extraction"
        )

        if (
            claim_path.exists()
            and evidence_path.exists()
            and list(claim_path.glob("*.csv"))
            and list(evidence_path.glob("*.csv"))
        ):
            logger.debug("Processed datasets already exist.")
            return {}

        datasets = self._load_persuade()

        for dataset_name, splits in datasets.items():
            if dataset_name == "claim_extraction":
                output_path = claim_path
            elif dataset_name == "evidence_extraction":
                output_path = evidence_path
            else:
                continue

            output_path.mkdir(parents=True, exist_ok=True)

            for subset_name, df in splits.items():
                file_path = output_path / f"{subset_name.value}.csv"

                df.to_csv(file_path, index=False)

                logger.debug(f"Saved {file_path}: {len(df)} rows")
        return {}

    def _create_claim_dataset(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create a dataset for claim extraction.

        Each discourse segment is classified as either a claim/position or
        a non-claim. Splitting is performed later by essay identifier to
        prevent data leakage.

        Args:
            df (pd.DataFrame): Raw PERSUADE dataframe.

        Returns:
            pd.DataFrame: Dataset containing sentences and claim labels.

        Raises:
            DatasetError: If required columns are missing or no claim
                extraction samples are created.
        """
        required_columns = {
            "essay_id_comp",
            "discourse_type",
            "discourse_text",
            "discourse_start",
        }

        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise DatasetError(f"Missing required columns: {missing_columns}")

        rows: list[dict[str, str | int]] = []

        claim_types = {
            "claim",
            "position",
        }

        for raw_essay_id, group_df in df.groupby("essay_id_comp"):
            essay_id = str(raw_essay_id)
            essay_df = group_df.sort_values("discourse_start")
            for _, row in essay_df.iterrows():
                text = row["discourse_text"]

                if pd.isna(text):
                    continue

                text = self._clean_sentence(str(text))
                if not text:
                    continue

                discourse_type = str(row["discourse_type"]).strip().lower()
                rows.append(
                    {
                        "essay_id_comp": str(essay_id),
                        "sentence": text,
                        "is_claim": int(discourse_type in claim_types),
                    }
                )
        result = pd.DataFrame(rows)

        if result.empty:
            raise DatasetError("No claim extraction samples were created.")
        return result[result["sentence"].str.split().str.len() >= MIN_CLAIM_WORDS]

    def _create_claim_evidence_pairs(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create claim-evidence pairs from PERSUADE annotations.

        Args:
            df (pd.DataFrame): Raw PERSUADE dataframe.

        Returns:
            pd.DataFrame: Dataset containing claim-evidence pairs.

        Raises:
            DatasetError: If required columns are missing.
        """
        required_columns = {
            "essay_id_comp",
            "discourse_type",
            "discourse_text",
            "discourse_start",
        }
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise DatasetError(f"Missing required columns: {missing_columns}")

        pairs: list[dict[str, str | int]] = []
        claim_types = {
            "claim",
            "position",
        }

        for raw_essay_id, group_df in df.groupby("essay_id_comp"):
            essay_id = str(raw_essay_id)
            current_claim: str | None = None

            for _, row in group_df.iterrows():
                discourse_type = str(row["discourse_type"]).strip().lower()
                text = row["discourse_text"]

                if pd.isna(text) or not str(text).strip():
                    continue
                text = self._clean_sentence(str(text))

                if discourse_type in claim_types:
                    current_claim = text

                elif discourse_type == "evidence":
                    if current_claim is None:
                        continue
                    pairs.append(
                        {
                            "essay_id_comp": essay_id,
                            "claim": current_claim,
                            "evidence": text,
                            "is_evidence": 1,
                        }
                    )
        result = pd.DataFrame(pairs)
        if result.empty:
            return result

        return result.drop_duplicates(
            subset=[
                "claim",
                "evidence",
            ],
            keep="first",
        )

    def _add_negative_pairs(
        self,
        positive_pairs: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate negative claim-evidence pairs.

        Negative examples are created by pairing claims with evidence from
        different arguments.

        Args:
            positive_pairs (pd.DataFrame): Dataframe containing positive
                claim-evidence pairs.

        Returns:
            pd.DataFrame: Dataframe containing positive and negative pairs.
        """
        negative_pairs = positive_pairs.copy()
        negative_pairs["claim"] = (
            negative_pairs["claim"]
            .sample(
                frac=XGBoostExtractorConfig.dataset.negative_sample_frac,
                random_state=XGBoostExtractorConfig.dataset.random_state,
            )
            .reset_index(drop=True)
        )

        negative_pairs["is_evidence"] = 0
        negative_pairs = negative_pairs[
            negative_pairs["claim"] != positive_pairs["claim"].to_numpy()
        ]
        return (
            pd.concat(
                [
                    positive_pairs,
                    negative_pairs,
                ],
                ignore_index=True,
            )
            .sample(
                frac=self.SAMPLE_ALL_RECORDS,
                random_state=XGBoostExtractorConfig.dataset.random_state,
            )
            .reset_index(drop=True)
        )

    def get_claim_split(
        self,
        split: SubsetName,
        max_samples: int | None = None,
    ) -> pd.DataFrame:
        """
        Get a claim extraction dataset split.

        Args:
            split (SubsetName): Dataset subset to retrieve.
            max_samples (int | None): Optional maximum number of rows to
                return.

        Returns:
            pd.DataFrame: Claim extraction dataframe.

        Raises:
            DatasetError: If the dataset split does not exist or does not
                contain the required columns.
        """
        dataset_path = (
            config.data_directory
            / config.preprocessed_dataset_subdirectory
            / "claim_extraction"
        )
        subset_file = dataset_path / f"{split.value}.csv"

        if not subset_file.exists():
            raise DatasetError(f"Dataset split does not exist: {subset_file}")
        df = pd.read_csv(subset_file)

        if "sentence" not in df.columns:
            raise DatasetError(
                "Claim extraction dataset must contain 'sentence' column."
            )
        df = filter_empty_rows(df, "sentence")

        if max_samples is not None:
            df = df.iloc[:max_samples]
        return df.reset_index(drop=True)

    def get_evidence_split(
        self,
        split: SubsetName,
        max_samples: int | None = None,
    ) -> pd.DataFrame:
        """
        Get an evidence extraction dataset split.

        Args:
            split (SubsetName): Dataset subset to retrieve.
            max_samples (int | None): Optional maximum number of rows to
                return.

        Returns:
            pd.DataFrame: Evidence extraction dataframe.

        Raises:
            DatasetError: If the dataset split does not exist or required
                columns are missing.
        """
        dataset_path = (
            config.data_directory
            / config.preprocessed_dataset_subdirectory
            / "evidence_extraction"
        )
        subset_file = dataset_path / f"{split.value}.csv"

        if not subset_file.exists():
            raise DatasetError(f"Dataset split does not exist: {subset_file}")
        df = pd.read_csv(subset_file)
        required_columns = {
            "claim",
            "evidence",
            "is_evidence",
        }

        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise DatasetError(f"Evidence dataset missing columns: {missing_columns}")
        df = df.dropna(
            subset=[
                "claim",
                "evidence",
            ]
        )
        df = df[
            (df["claim"].astype(str).str.strip() != "")
            & (df["evidence"].astype(str).str.strip() != "")
        ]

        if max_samples is not None:
            df = df.iloc[:max_samples]
        return df.reset_index(drop=True)

"""Module with an argument detection dataset."""

import asyncio
import shutil
import zipfile
from pathlib import Path
from typing import override

import httpx
import pandas as pd
import py7zr
from loguru import logger
from sklearn.model_selection import train_test_split

from src.configuration import config
from src.data_models.abstract.dataset import Dataset
from src.data_models.data_models import SubsetName
from src.utils.dataset import to_raw_dataset_path


class ArgumentDetectionDataset(Dataset):
    """Dataset for argument detection."""

    @staticmethod
    def _clean_sentence(sentence: str) -> str:
        """Clean sentence by removing newlines and normalizing whitespace."""
        # Replace all newlines with spaces.
        cleaned = sentence.replace("\n", " ").replace("\r", " ")
        # Replace multiple consecutive spaces with a single space.
        return " ".join(cleaned.split())

    def __init__(self) -> None:
        """Define raw databases to be transformed."""
        # PubMed RCT: https://github.com/Franck-Dernoncourt/pubmed-rct
        # Persuade 1.0:
        # https://www.kaggle.com/datasets/julesking/tla-lab-persuade-dataset?select=persuade2_train_srctexts.csv

        # Data:
        # Add Persuade 2.0 exists: https://github.com/scrosseye/persuade_corpus_2.0 and
        # Remove Persuade 1.0
        # Add AAE: https://tudatalib.ulb.tu-darmstadt.de/items/9177c48c-8bd5-4881-9cb4-0632b5941464
        self._datasets = {
            to_raw_dataset_path("persuade"): [
                (
                    "https://www.kaggle.com/api/v1/datasets/download/"
                    "julesking/tla-lab-persuade-dataset"
                ),
            ],
            to_raw_dataset_path("pubmed-rct"): [
                (
                    "https://raw.githubusercontent.com/Franck-Dernoncourt/pubmed-rct/"
                    "17ed2cb0590decfca0266add0c76f254f67232b4/PubMed_200k_RCT/train.7z"
                ),
                (
                    "https://raw.githubusercontent.com/Franck-Dernoncourt/pubmed-rct/"
                    "17ed2cb0590decfca0266add0c76f254f67232b4/PubMed_200k_RCT/dev.txt"
                ),
                (
                    "https://raw.githubusercontent.com/Franck-Dernoncourt/pubmed-rct/"
                    "17ed2cb0590decfca0266add0c76f254f67232b4/PubMed_200k_RCT/test.txt"
                ),
            ],
        }

    @override
    async def prepare(self) -> Path:
        await self._download_raw_datasets_if_missing()
        datasets = self._transform()
        return self._merge_datasets(datasets)

    @override
    def clear(self) -> None:
        for dataset in self._datasets:
            shutil.rmtree(dataset)

    async def _download_raw_datasets_if_missing(self) -> None:
        """Download raw datasets if missing."""
        logger.debug("Starting raw datasets download check")
        tasks = []

        async with httpx.AsyncClient() as client:
            for local_path, urls in self._datasets.items():
                local_path.mkdir(parents=True, exist_ok=True)

                for url in urls:
                    file_name = url.rsplit("/", maxsplit=1)[-1]
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
        merged_subsets: dict[SubsetName, list] = {
            SubsetName.TRAINING: [],
            SubsetName.VALIDATION: [],
            SubsetName.TESTING: [],
        }

        # Collect all dataframes for each subset from all datasets
        for subsets in extracted_data.values():
            for subset_name, df in subsets.items():
                if len(df) > 0:  # Only add non-empty dataframes
                    merged_subsets[subset_name].append(df)

        # Merge and shuffle each subset
        merged_data = {}
        for subset_name, dfs in merged_subsets.items():
            if dfs:
                # Concatenate all dataframes for this subset
                merged_df = pd.concat(dfs, ignore_index=True)
                # Shuffle the data while maintaining stratification
                merged_df = merged_df.sample(frac=1, random_state=42).reset_index(
                    drop=True
                )
                merged_data[subset_name] = merged_df
            else:
                merged_data[subset_name] = pd.DataFrame(
                    {"sentence": [], "is_argument": []}
                )

        # Save merged dataset to a processed dataset location.
        processed_dataset_path = (
            config.data_directory
            / config.preprocessed_dataset_subdirectory
            / "argument_detection"
        )
        processed_dataset_path.mkdir(parents=True, exist_ok=True)

        # Save each subset as a CSV file
        for subset_name, df in merged_data.items():
            subset_file = processed_dataset_path / f"{subset_name.value}.csv"
            df.to_csv(subset_file, index=False)

        self._log_dataset_statistics(merged_data, processed_dataset_path)
        return processed_dataset_path

    def _log_dataset_statistics(
        self, merged_data: dict[SubsetName, pd.DataFrame], processed_dataset_path: Path
    ) -> None:
        total_sentences = 0
        total_evidence = 0
        for subset_name, df in merged_data.items():
            evidence_count = df["is_argument"].sum()
            evidence_pct = (evidence_count / len(df)) * 100 if len(df) > 0 else 0
            total_sentences += len(df)
            total_evidence += evidence_count
            logger.debug(f"{subset_name.value}: {len(df)} sentences")
            logger.debug(f"  Evidence: {evidence_count} ({evidence_pct:.1f}%)")
            logger.debug(
                f"  Non-Evidence: {(~df['is_argument']).sum()} "
                f"({100 - evidence_pct:.1f}%)"
            )

        overall_evidence_pct = (
            (total_evidence / total_sentences) * 100 if total_sentences > 0 else 0
        )
        logger.debug(f"Total: {total_sentences} sentences")
        logger.debug(
            f"Overall Evidence: {total_evidence} ({overall_evidence_pct:.1f}%)"
        )
        logger.debug(f"Dataset saved to: {processed_dataset_path}")

    def _load_pubmed_rct(self) -> dict[SubsetName, pd.DataFrame]:
        dataset_path = to_raw_dataset_path("pubmed-rct")
        archive_path = dataset_path / "train.7z"

        # Extract the 7z file.
        if archive_path.exists():
            with py7zr.SevenZipFile(archive_path, "r") as archive:
                archive.extractall(path=dataset_path)

        # PubMed RCT has several classes.
        # Classes with True are considered arguments.
        mapping = {
            "BACKGROUND": False,
            "OBJECTIVE": False,
            "METHODS": False,
            "CONCLUSIONS": True,
            "RESULTS": True,
        }

        def _convert_pubmed_rct_to_final_format(file_path: Path) -> pd.DataFrame:
            sentences = []
            is_arguments = []
            parts_per_line = 2

            with file_path.open("r", encoding="utf-8") as f:
                for _line in f:
                    line = _line.strip()

                    # Skip empty lines and PMID headers.
                    if not line or line.startswith("###"):
                        continue

                    # Parse tab-separated format: SECTION_CLASS\tsentence
                    parts = line.split("\t", maxsplit=1)
                    if len(parts) != parts_per_line:
                        continue  # Skip malformed lines.

                    section_class, sentence = parts
                    section_class = section_class.strip()
                    sentence = self._clean_sentence(sentence)

                    # Map section class to is_argument label
                    is_arg = mapping.get(section_class, False)

                    sentences.append(sentence)
                    is_arguments.append(is_arg)

            return pd.DataFrame(
                {
                    "sentence": sentences,
                    "is_argument": is_arguments,
                }
            )

        train_path = dataset_path / "train.txt"
        val_path = dataset_path / "dev.txt"
        test_path = dataset_path / "test.txt"

        return {
            SubsetName.TRAINING: _convert_pubmed_rct_to_final_format(train_path),
            SubsetName.VALIDATION: _convert_pubmed_rct_to_final_format(val_path),
            SubsetName.TESTING: _convert_pubmed_rct_to_final_format(test_path),
        }

    def _load_persuade(self) -> dict[SubsetName, pd.DataFrame]:
        dataset_directory = to_raw_dataset_path("persuade")
        zip_file = dataset_directory / "download.zip"

        # Unzip file if it exists.
        if zip_file.exists() and zipfile.is_zipfile(zip_file):
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(dataset_directory)

        # Read the CSV file
        csv_file = dataset_directory / "persuade2_train_srctexts.csv"
        if not csv_file.exists():
            return {
                SubsetName.TRAINING: pd.DataFrame({"sentence": [], "is_argument": []}),
                SubsetName.VALIDATION: pd.DataFrame(
                    {"sentence": [], "is_argument": []}
                ),
                SubsetName.TESTING: pd.DataFrame({"sentence": [], "is_argument": []}),
            }

        # Read CSV and convert to final format.
        df = pd.read_csv(csv_file, low_memory=False)

        # Convert to final format: sentences and is_argument labels.
        # Evidence = True, other discourse types = False
        sentences = []
        is_arguments = []

        for _, row in df.iterrows():
            # Skip rows with empty discourse_text.
            if (
                pd.isna(row.get("discourse_text"))
                or not str(row.get("discourse_text")).strip()
            ):
                continue

            sentence = self._clean_sentence(str(row.get("discourse_text", "")))
            discourse_type = str(row.get("discourse_type", "")).strip()

            # Map discourse_type to is_argument: Evidence = True, others = False
            is_arg = discourse_type.lower() == "evidence"

            sentences.append(sentence)
            is_arguments.append(is_arg)

        df = pd.DataFrame(
            {
                "sentence": sentences,
                "is_argument": is_arguments,
            }
        )

        train_df, val_df, test_df = self._split_with_stratification(df)
        return {
            SubsetName.TRAINING: train_df.reset_index(drop=True),
            SubsetName.VALIDATION: val_df.reset_index(drop=True),
            SubsetName.TESTING: test_df.reset_index(drop=True),
        }

    def _split_with_stratification(
        self, final_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Split into training (70%), validation (15%), and testing (15%) with
        # stratification.
        train_df, temp_df = train_test_split(
            final_df,
            test_size=0.3,
            stratify=final_df["is_argument"],
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            stratify=temp_df["is_argument"],
        )

        return train_df, val_df, test_df

    def _transform(self) -> dict[Path, dict[SubsetName, pd.DataFrame]]:
        pubmed_rct = self._load_pubmed_rct()
        persuade = self._load_persuade()
        return {
            to_raw_dataset_path("pubmed-rct"): pubmed_rct,
            to_raw_dataset_path("persuade"): persuade,
        }

"""Module with a dataset utilities useful in multiple modules."""

from pathlib import Path

import pandas as pd

from src.configuration import config


def to_raw_dataset_path(raw_dataset_name: str) -> Path:
    """
    Convert a sole name to path of a raw (unprocessed) dataset.

    This function does NOT guarantee the dataset will exist under this path.
    Only that it will be deterministic and conformat with the configuration.

    Args:
        raw_dataset_name (str): Name of the raw dataset.

    Returns:
        Path: Path where a raw dataset should be.
    """
    raw_dataset_name = raw_dataset_name.replace(" ", "_").replace("-", "_")
    return config.data_directory / config.raw_dataset_subdirectory / raw_dataset_name


def filter_empty_rows(df: pd.DataFrame, columns: str | list[str]) -> pd.DataFrame:
    """
    Filter out rows where specified columns are NaN or contain only whitespace.

    Args:
        df: Input pandas DataFrame.
        columns: A single column name or a list of column names to validate.

    Returns:
        A shallow copy of the DataFrame with empty rows removed.
    """
    cols = [columns] if isinstance(columns, str) else columns

    df_cleaned = df.dropna(subset=cols)

    for col in cols:
        df_cleaned = df_cleaned[df_cleaned[col].astype(str).str.strip() != ""]

    return df_cleaned.copy()

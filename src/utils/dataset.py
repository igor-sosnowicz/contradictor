"""Module with a dataset utilities useful in multiple modules."""

import pandas as pd


def filter_empty_rows(df: pd.DataFrame, columns: str | list[str]) -> pd.DataFrame:
    """
    Filter out rows where specified columns are empty or contain only whitespace.

    Args:
        df (pd.DataFrame): Input pandas DataFrame.
        columns (str | list[str]): Column name or list of column names to
            validate.

    Returns:
        pd.DataFrame: A copy of the DataFrame with empty rows removed.
    """
    cols = [columns] if isinstance(columns, str) else columns

    df_cleaned = df.dropna(subset=cols)

    for col in cols:
        df_cleaned = df_cleaned[df_cleaned[col].astype(str).str.strip() != ""]

    return df_cleaned.copy()

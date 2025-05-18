"""Contains utility functions for data manipulation and augmentation"""

import numpy as np
import pandas as pd


def introduce_missing_values(
    df: pd.DataFrame, percentage: float
) -> pd.DataFrame:
    """Introduce random missing values into the dataset"""
    if percentage <= 0:
        return df

    # Make a copy to avoid modifying the original
    df_copy = df.copy()

    # Calculate how many values to set as NaN
    total_cells = df_copy.size
    cells_to_nullify = int(total_cells * percentage / 100)

    # Get indices of all cells
    indices = [
        (i, j)
        for i in range(len(df_copy))
        for j in range(len(df_copy.columns))
    ]

    # Randomly select cells to nullify
    null_indices = np.random.choice(
        len(indices), cells_to_nullify, replace=False
    )

    # Set selected cells to NaN
    for idx in null_indices:
        i, j = indices[idx]
        df_copy.iloc[i, j] = np.nan

    return df_copy


def introduce_duplicates(df: pd.DataFrame, percentage: float) -> pd.DataFrame:
    """Introduce duplicate records into the dataset"""
    if percentage <= 0:
        return df

    # Make a copy to avoid modifying the original
    df_copy = df.copy()

    # Calculate how many rows to duplicate
    rows_to_duplicate = int(len(df_copy) * percentage / 100)

    # Randomly select rows to duplicate
    duplicate_indices = np.random.choice(
        len(df_copy), rows_to_duplicate, replace=False
    )

    # Create duplicates
    duplicates = df_copy.iloc[duplicate_indices].copy()

    # Append duplicates to the dataset
    return pd.concat([df_copy, duplicates], ignore_index=True)


def introduce_outliers(df: pd.DataFrame, percentage: float) -> pd.DataFrame:
    """Introduce outliers into numeric columns of the dataset"""
    if percentage <= 0:
        return df

    # Make a copy to avoid modifying the original
    df_copy = df.copy()

    # Get numeric columns
    numeric_columns = df_copy.select_dtypes(include=["number"]).columns

    if len(numeric_columns) == 0:
        return df_copy

    # Calculate how many outliers to introduce
    rows = len(df_copy)
    cols = len(numeric_columns)
    cells_to_modify = int(rows * cols * percentage / 100)

    # Randomly select cells to modify
    row_indices = np.random.choice(rows, cells_to_modify, replace=True)
    col_indices = np.random.choice(
        len(numeric_columns), cells_to_modify, replace=True
    )

    # Introduce outliers
    for i, j in zip(row_indices, col_indices):
        col = numeric_columns[j]

        # Calculate mean and std for this column
        mean = df_copy[col].mean()
        std = df_copy[col].std()

        # Set value to be 5-10 std away from mean (outlier)
        outlier_factor = np.random.uniform(5, 10) * np.random.choice([-1, 1])
        df_copy.iloc[i, df_copy.columns.get_loc(col)] = (
            mean + outlier_factor * std
        )

    return df_copy

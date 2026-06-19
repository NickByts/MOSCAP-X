"""Validation, cleaning, and unit conversion for MOSCAP-X datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .constants import (
        CAP_UNIT_FACTORS,
        REQUIRED_STANDARD_COLUMNS,
        STANDARD_CAPACITANCE_COLUMN,
        STANDARD_CONDUCTANCE_COLUMN,
        STANDARD_VOLTAGE_COLUMN,
    )
except ImportError:
    from constants import (
        CAP_UNIT_FACTORS,
        REQUIRED_STANDARD_COLUMNS,
        STANDARD_CAPACITANCE_COLUMN,
        STANDARD_CONDUCTANCE_COLUMN,
        STANDARD_VOLTAGE_COLUMN,
    )


def validate_data(dataframe: pd.DataFrame) -> list[str]:
    """Validate measurement data and return non-fatal warning messages."""
    warnings: list[str] = []

    if dataframe.empty:
        warnings.append(
            "The dataframe is empty; upload a file with measurement rows."
        )
        return warnings

    missing_columns = [
        column
        for column in REQUIRED_STANDARD_COLUMNS
        if column not in dataframe.columns
    ]
    if missing_columns:
        warnings.append(
            "Missing standardized column(s): " + ", ".join(missing_columns)
        )
        return warnings

    analysis_columns = _analysis_columns(dataframe)

    missing_value_count = int(dataframe[analysis_columns].isna().sum().sum())
    if missing_value_count > 0:
        warnings.append(
            f"Found {missing_value_count} missing value(s) in measurement data."
        )

    duplicate_count = int(dataframe.duplicated().sum())
    if duplicate_count > 0:
        warnings.append(f"Found {duplicate_count} duplicate row(s).")

    for column in analysis_columns:
        numeric_values = pd.to_numeric(dataframe[column], errors="coerce")
        non_numeric_count = int(
            (numeric_values.isna() & dataframe[column].notna()).sum()
        )
        if non_numeric_count > 0:
            warnings.append(
                f"Column '{column}' contains {non_numeric_count} "
                "non-numeric value(s)."
            )

    capacitance = pd.to_numeric(
        dataframe[STANDARD_CAPACITANCE_COLUMN],
        errors="coerce",
    )
    negative_capacitance_count = int((capacitance < 0).sum())
    if negative_capacitance_count > 0:
        warnings.append(
            "Found "
            f"{negative_capacitance_count} negative capacitance value(s)."
        )

    return warnings


def clean_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Clean measurement data by coercing numeric values and sorting voltage."""
    _require_columns(dataframe, REQUIRED_STANDARD_COLUMNS)

    columns = _analysis_columns(dataframe)
    cleaned = dataframe.loc[:, columns].copy()

    for column in columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.dropna(subset=columns)
    cleaned = cleaned.drop_duplicates()
    cleaned = cleaned.sort_values(by=STANDARD_VOLTAGE_COLUMN, kind="mergesort")
    cleaned = cleaned.reset_index(drop=True)

    return cleaned


def convert_capacitance_units(
    dataframe: pd.DataFrame,
    unit: str,
) -> pd.DataFrame:
    """Convert capacitance values from the selected unit into Farads."""
    if unit not in CAP_UNIT_FACTORS:
        supported_units = ", ".join(CAP_UNIT_FACTORS)
        raise ValueError(
            f"Unsupported capacitance unit '{unit}'. "
            f"Supported units: {supported_units}."
        )

    _require_columns(dataframe, (STANDARD_CAPACITANCE_COLUMN,))

    converted = dataframe.copy()
    converted[STANDARD_CAPACITANCE_COLUMN] = pd.to_numeric(
        converted[STANDARD_CAPACITANCE_COLUMN],
        errors="coerce",
    ) * CAP_UNIT_FACTORS[unit]

    return converted


def _analysis_columns(dataframe: pd.DataFrame) -> list[str]:
    columns = [
        STANDARD_VOLTAGE_COLUMN,
        STANDARD_CAPACITANCE_COLUMN,
    ]

    if STANDARD_CONDUCTANCE_COLUMN in dataframe.columns:
        columns.append(STANDARD_CONDUCTANCE_COLUMN)

    return columns


def _require_columns(
    dataframe: pd.DataFrame,
    required_columns: tuple[str, ...],
) -> None:
    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing required column(s): " + ", ".join(missing_columns)
        )

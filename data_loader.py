"""Data import and column standardization utilities for MOSCAP-X."""

from __future__ import annotations

from pathlib import Path
from typing import Any, IO, Iterable

import pandas as pd

try:
    from .constants import (
        CAPACITANCE_COLUMN_NAMES,
        CONDUCTANCE_COLUMN_NAMES,
        STANDARD_CAPACITANCE_COLUMN,
        STANDARD_CONDUCTANCE_COLUMN,
        STANDARD_VOLTAGE_COLUMN,
        SUPPORTED_CSV_EXTENSIONS,
        SUPPORTED_EXCEL_EXTENSIONS,
        VOLTAGE_COLUMN_NAMES,
    )
except ImportError:
    from constants import (
        CAPACITANCE_COLUMN_NAMES,
        CONDUCTANCE_COLUMN_NAMES,
        STANDARD_CAPACITANCE_COLUMN,
        STANDARD_CONDUCTANCE_COLUMN,
        STANDARD_VOLTAGE_COLUMN,
        SUPPORTED_CSV_EXTENSIONS,
        SUPPORTED_EXCEL_EXTENSIONS,
        VOLTAGE_COLUMN_NAMES,
    )

ReadableFile = str | Path | IO[str] | IO[bytes]


def load_csv(filepath: ReadableFile) -> pd.DataFrame:
    """Load a CSV file and standardize MOS capacitor measurement columns."""
    try:
        dataframe = pd.read_csv(filepath)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The CSV file is empty.") from exc
    except Exception as exc:
        raise ValueError(f"Unable to read CSV file: {exc}") from exc

    return _standardize_columns(dataframe)


def load_excel(filepath: ReadableFile) -> pd.DataFrame:
    """Load an Excel workbook and standardize MOS capacitor columns."""
    try:
        dataframe = pd.read_excel(filepath, engine="openpyxl")
    except ValueError as exc:
        raise ValueError(
            "Unable to read Excel file with openpyxl. Use a supported "
            ".xlsx, .xlsm, .xltx, or .xltm workbook."
        ) from exc
    except Exception as exc:
        raise ValueError(f"Unable to read Excel file: {exc}") from exc

    return _standardize_columns(dataframe)


def load_data(uploaded_file: Any) -> pd.DataFrame:
    """Load an uploaded CSV or Excel file and standardize its columns."""
    filename = str(getattr(uploaded_file, "name", ""))
    suffix = Path(filename).suffix.lower()

    if not suffix:
        raise ValueError("Unable to determine file type from the uploaded file.")

    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)

    if suffix in SUPPORTED_CSV_EXTENSIONS:
        return load_csv(uploaded_file)

    if suffix in SUPPORTED_EXCEL_EXTENSIONS:
        return load_excel(uploaded_file)

    supported = ", ".join(
        sorted(SUPPORTED_CSV_EXTENSIONS + SUPPORTED_EXCEL_EXTENSIONS)
    )
    raise ValueError(
        f"Unsupported file type '{suffix}'. Supported file types: {supported}."
    )


def _standardize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    voltage_column = _find_column(dataframe.columns, VOLTAGE_COLUMN_NAMES)
    capacitance_column = _find_column(
        dataframe.columns,
        CAPACITANCE_COLUMN_NAMES,
    )
    conductance_column = _find_column(
        dataframe.columns,
        CONDUCTANCE_COLUMN_NAMES,
    )

    missing: list[str] = []
    if voltage_column is None:
        missing.append(
            "Voltage "
            f"(accepted names: {', '.join(VOLTAGE_COLUMN_NAMES)})"
        )
    if capacitance_column is None:
        missing.append(
            "Capacitance "
            f"(accepted names: {', '.join(CAPACITANCE_COLUMN_NAMES)})"
        )

    if missing:
        available_columns = ", ".join(map(str, dataframe.columns))
        raise ValueError(
            "Missing required column(s): "
            f"{'; '.join(missing)}. Available columns: {available_columns}."
        )

    selected_columns = [voltage_column, capacitance_column]
    rename_map = {
        voltage_column: STANDARD_VOLTAGE_COLUMN,
        capacitance_column: STANDARD_CAPACITANCE_COLUMN,
    }

    if conductance_column is not None:
        selected_columns.append(conductance_column)
        rename_map[conductance_column] = STANDARD_CONDUCTANCE_COLUMN

    standardized = dataframe.loc[:, selected_columns].rename(columns=rename_map)
    return standardized


def _find_column(
    columns: Iterable[Any],
    accepted_names: tuple[str, ...],
) -> Any | None:
    normalized_lookup: dict[str, Any] = {}
    for column in columns:
        normalized_lookup.setdefault(_normalize_column_name(column), column)

    for accepted_name in accepted_names:
        normalized_name = _normalize_column_name(accepted_name)
        if normalized_name in normalized_lookup:
            return normalized_lookup[normalized_name]

    return None


def _normalize_column_name(column_name: Any) -> str:
    return "".join(str(column_name).strip().lower().replace("_", " ").split())

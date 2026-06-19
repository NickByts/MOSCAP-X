"""Statistical summaries for MOSCAP-X measurement data."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .constants import (
        STANDARD_CAPACITANCE_COLUMN,
        STANDARD_VOLTAGE_COLUMN,
    )
except ImportError:
    from constants import (
        STANDARD_CAPACITANCE_COLUMN,
        STANDARD_VOLTAGE_COLUMN,
    )

StatisticValue = int | float


def calculate_statistics(dataframe: pd.DataFrame) -> dict[str, StatisticValue]:
    """Calculate key voltage and capacitance statistics for cleaned data."""
    if dataframe.empty:
        raise ValueError("Cannot calculate statistics for an empty dataframe.")

    required_columns = (
        STANDARD_VOLTAGE_COLUMN,
        STANDARD_CAPACITANCE_COLUMN,
    )
    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing required column(s): " + ", ".join(missing_columns)
        )

    voltage = pd.to_numeric(
        dataframe[STANDARD_VOLTAGE_COLUMN],
        errors="coerce",
    ).dropna()
    capacitance = pd.to_numeric(
        dataframe[STANDARD_CAPACITANCE_COLUMN],
        errors="coerce",
    ).dropna()

    if voltage.empty or capacitance.empty:
        raise ValueError(
            "Statistics require numeric voltage and capacitance values."
        )

    capacitance_array = capacitance.to_numpy(dtype=float)

    return {
        "total_points": int(len(dataframe)),
        "voltage_min": float(voltage.min()),
        "voltage_max": float(voltage.max()),
        "capacitance_min": float(capacitance.min()),
        "capacitance_max": float(capacitance.max()),
        "capacitance_mean": float(capacitance.mean()),
        "capacitance_std": float(np.std(capacitance_array, ddof=0)),
    }

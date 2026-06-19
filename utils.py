"""Shared utility functions for the MOSCAP-X Streamlit application."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

try:
    from .constants import (
        NORMALIZED_CAPACITANCE_COLUMN,
        STANDARD_CAPACITANCE_COLUMN,
    )
except ImportError:
    from constants import (
        NORMALIZED_CAPACITANCE_COLUMN,
        STANDARD_CAPACITANCE_COLUMN,
    )


def add_normalized_capacitance(
    dataframe: pd.DataFrame,
    device_area_cm2: float,
) -> pd.DataFrame:
    """Add area-normalized capacitance in F/cm^2 to a dataframe copy."""
    if device_area_cm2 <= 0.0:
        raise ValueError("Device area must be greater than zero.")

    if STANDARD_CAPACITANCE_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Missing required column '{STANDARD_CAPACITANCE_COLUMN}'."
        )

    normalized = dataframe.copy()
    normalized[NORMALIZED_CAPACITANCE_COLUMN] = pd.to_numeric(
        normalized[STANDARD_CAPACITANCE_COLUMN],
        errors="coerce",
    ) / device_area_cm2

    return normalized


def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    """Serialize a dataframe to UTF-8 CSV bytes without the index."""
    return dataframe.to_csv(index=False).encode("utf-8")


def figure_to_png_bytes(figure: Figure) -> bytes:
    """Serialize a Matplotlib figure into PNG bytes."""
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


def format_scientific(value: int | float, precision: int = 3) -> str:
    """Format numeric values compactly for Streamlit metric displays."""
    numeric_value = float(value)

    if not np.isfinite(numeric_value):
        return "N/A"

    if numeric_value == 0.0:
        return "0"

    if abs(numeric_value) >= 1.0e4 or abs(numeric_value) < 1.0e-3:
        return f"{numeric_value:.{precision}e}"

    return f"{numeric_value:.{precision}g}"

"""Linear regression engine for MOSCAP-X Phase 1B."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import linregress


@dataclass(frozen=True)
class FitResult:
    """Container for linear fit parameters."""

    slope: float
    intercept: float
    r2: float


def perform_linear_fit(
    selected_voltage: Sequence[float] | np.ndarray,
    selected_inverse_c2: Sequence[float] | np.ndarray,
) -> FitResult:
    """Fit a line to selected 1/C^2-V data using scipy.stats.linregress."""
    voltage = _as_float_array(selected_voltage, "selected_voltage")
    inverse_c2 = _as_float_array(selected_inverse_c2, "selected_inverse_c2")
    _validate_regression_inputs(voltage, inverse_c2)

    try:
        result = linregress(voltage, inverse_c2)
    except ValueError as exc:
        raise ValueError(f"Linear regression failed: {exc}") from exc

    slope = float(result.slope)
    intercept = float(result.intercept)
    r_value = float(result.rvalue)
    r2 = r_value * r_value

    if not np.all(np.isfinite([slope, intercept, r2])):
        raise ValueError("Linear regression returned non-finite results.")

    return FitResult(slope=slope, intercept=intercept, r2=float(r2))


def _as_float_array(
    values: Sequence[float] | np.ndarray,
    name: str,
) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc

    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")

    return array


def _validate_regression_inputs(
    voltage: np.ndarray,
    inverse_c2: np.ndarray,
) -> None:
    if voltage.size != inverse_c2.size:
        raise ValueError(
            "selected_voltage and selected_inverse_c2 must have equal length."
        )

    if voltage.size < 2:
        raise ValueError("At least two points are required for regression.")

    if not np.all(np.isfinite(voltage)):
        raise ValueError("selected_voltage contains NaN or infinite values.")

    if not np.all(np.isfinite(inverse_c2)):
        raise ValueError("selected_inverse_c2 contains NaN or infinite values.")

    if float(np.ptp(voltage)) == 0.0:
        raise ValueError("Regression requires at least two distinct voltages.")

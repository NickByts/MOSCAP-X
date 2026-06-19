"""Automatic depletion-region detection for MOSCAP-X Phase 1B."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

try:
    from .linear_regression import FitResult, perform_linear_fit
    from .region_classifier import classify_regions
except ImportError:
    from linear_regression import FitResult, perform_linear_fit
    from region_classifier import classify_regions

DEFAULT_WINDOW_PERCENTAGES: tuple[float, ...] = (
    0.05,   # 5%
    0.08,   # 8%
    0.10,   # 10%
    0.12,   # 12%
    0.15,   # 15%
    0.20,   # 20%
)
DEFAULT_MIN_R2 = 0.98
R2_TIE_TOLERANCE = 1.0e-6


@dataclass(frozen=True)
class LinearRegionResult:
    """Detected linear depletion-region window and regression parameters."""

    start_index: int
    end_index: int
    slope: float
    intercept: float
    r2: float

def _generate_window_sizes(
    point_count: int,
) -> tuple[int, ...]:
    """
    Generate candidate window sizes as percentages of dataset length.
    """

    window_sizes = {
        max(10, int(point_count * percentage))
        for percentage in DEFAULT_WINDOW_PERCENTAGES
    }

    return tuple(sorted(window_sizes))

def _smooth_inverse_c2(
    inverse_c2: np.ndarray,
) -> np.ndarray:
    """
    Smooth noisy 1/C² data before region detection.
    """

    point_count = len(inverse_c2)

    if point_count < 31:
        return inverse_c2

    return savgol_filter(
        inverse_c2,
        window_length=31,
        polyorder=3,
    )

def _calculate_local_slope(
    voltage: np.ndarray,
    inverse_c2: np.ndarray,
) -> np.ndarray:
    """
    Numerical derivative d(1/C²)/dV
    """

    return np.gradient(
        inverse_c2,
        voltage,
    )


def detect_linear_region(
    voltage_array,
    inverse_capacitance_squared,
    min_r2: float = DEFAULT_MIN_R2,
) -> LinearRegionResult:
    """Detect the most linear depletion-region segment of a 1/C^2-V curve."""
    voltage = _as_valid_array(voltage_array, "voltage_array")
    inverse_c2 = _as_valid_array(
        inverse_capacitance_squared,
        "inverse_capacitance_squared",   
    )
    inverse_c2 = _smooth_inverse_c2(inverse_c2)

    _validate_detection_inputs(voltage, inverse_c2, min_r2)

    boundaries = classify_regions(
        voltage,
        inverse_c2,
    )
    print(
        "ACC_END =", boundaries.accumulation_end_index,
        "DEP_START =", boundaries.depletion_start_index,
        "DEP_END =", boundaries.depletion_end_index,
        "INV_START =", boundaries.inversion_start_index,
    )

    depletion_start = boundaries.depletion_start_index
    depletion_end = boundaries.depletion_end_index

    depletion_voltage = voltage[
        depletion_start:depletion_end + 1
    ]

    depletion_inverse_c2 = inverse_c2[
        depletion_start:depletion_end + 1
    ]

    candidate_sizes = _generate_window_sizes(
        len(depletion_voltage)
    )

    local_slope = _calculate_local_slope(
        depletion_voltage,
        depletion_inverse_c2,
    )

    best_result: LinearRegionResult | None = None
    best_stability = np.inf

    for window_size in candidate_sizes:
        for start_index in range(0, len(depletion_voltage) - window_size + 1,):
            end_index = start_index + window_size
            fit_result = _fit_window(
                depletion_voltage[start_index:end_index],
                depletion_inverse_c2[start_index:end_index],
            )
            window_slope = local_slope[start_index:end_index]

            stability = _slope_stability(
                window_slope
            )

            if fit_result is None or fit_result.r2 < min_r2:
                continue

            candidate = LinearRegionResult(
                start_index=(
                    depletion_start
                    + start_index
                ),
                end_index=(
                    depletion_start
                    + end_index
                    - 1
                ),
                slope=fit_result.slope,
                intercept=fit_result.intercept,
                r2=fit_result.r2,
            )
            if stability < best_stability:

                best_result = candidate

                best_stability = stability

    if best_result is None:
        raise ValueError(
            "No linear depletion-region window met the R^2 threshold "
            f"of {min_r2:.4f}."
        )

    return best_result

def fit_voltage_range(
    voltage_array,
    inverse_capacitance_squared,
    start_voltage: float,
    end_voltage: float,
) -> LinearRegionResult:
    """
    Search for the best linear region only inside a user-selected
    voltage range.
    """

    if start_voltage >= end_voltage:
        raise ValueError(
            "Start voltage must be smaller than end voltage."
        )

    voltage = _as_valid_array(
        voltage_array,
        "voltage_array",
    )

    inverse_c2 = _as_valid_array(
        inverse_capacitance_squared,
        "inverse_capacitance_squared",
    )

    mask = (
        (voltage >= start_voltage)
        &
        (voltage <= end_voltage)
    )

    selected_indices = np.where(mask)[0]

    if len(selected_indices) < 10:
        raise ValueError(
            "Selected voltage range contains too few points."
        )

    selected_voltage = voltage[mask]
    selected_inverse_c2 = inverse_c2[mask]

    local_region = detect_linear_region(
        selected_voltage,
        selected_inverse_c2,
    )

    global_start = (
        selected_indices[0]
        + local_region.start_index
    )

    global_end = (
        selected_indices[0]
        + local_region.end_index
    )

    return LinearRegionResult(
        start_index=int(global_start),
        end_index=int(global_end),
        slope=local_region.slope,
        intercept=local_region.intercept,
        r2=local_region.r2,
    )


def _fit_window(voltage: np.ndarray, inverse_c2: np.ndarray) -> FitResult | None:
    try:
        return perform_linear_fit(voltage, inverse_c2)
    except ValueError:
        return None


def _is_better_candidate(
    candidate: LinearRegionResult,
    current_best: LinearRegionResult | None,
    candidate_length: int,
    current_best_length: int,
) -> bool:
    if current_best is None:
        return True

    if candidate.r2 > current_best.r2 + R2_TIE_TOLERANCE:
        return True

    similar_r2 = abs(candidate.r2 - current_best.r2) <= R2_TIE_TOLERANCE
    return similar_r2 and candidate_length > current_best_length


def _as_valid_array(
    values: Sequence[float] | np.ndarray,
    name: str,
) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc

    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")

    return array

def _slope_stability(
    slope_values: np.ndarray,
) -> float:
    """
    Measures how constant the local slope is.

    Smaller value = more linear region.
    """

    mean_slope = np.mean(
        np.abs(slope_values)
    )

    if mean_slope == 0:
        return np.inf

    return (
        np.std(slope_values)
        / mean_slope
    )

def _validate_detection_inputs(
    voltage: np.ndarray,
    inverse_c2: np.ndarray,
    min_r2: float,
) -> None:
    if voltage.size != inverse_c2.size:
        raise ValueError(
            "voltage_array and inverse_capacitance_squared must have "
            "equal length."
        )

    if not np.isfinite(min_r2) or min_r2 < 0.0 or min_r2 > 1.0:
        raise ValueError("min_r2 must be a finite value between 0 and 1.")


# def _eligible_window_sizes(
#     window_sizes: Sequence[int],
#     point_count: int,
# ) -> tuple[int, ...]:
#     unique_sizes = sorted({int(size) for size in window_sizes})
#     eligible_sizes = tuple(
#         size for size in unique_sizes if 2 <= size <= point_count
#     )

#     if not eligible_sizes:
#         raise ValueError(
#             "Insufficient points for linear-region detection. At least "
#             f"{min(DEFAULT_WINDOW_PERCENTAGES) * 100:.0f}% of the dataset is required."
#         )

#     return eligible_sizes

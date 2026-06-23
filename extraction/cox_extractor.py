"""Oxide capacitance extraction for MOSCAP-X Phase 2."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

try:
    from .phase2_validation import validate_array, validate_unit
except ImportError:
    from phase2_validation import validate_array, validate_unit


@dataclass(frozen=True)
class CoxResult:
    """Oxide capacitance estimate and accumulation-region quality metrics."""

    cox: float
    confidence: float
    points_used: int


def calculate_cox(
    capacitance_array: Sequence[float] | np.ndarray,
    top_fraction: float = 0.10,
    *,
    capacitance_unit: str = "F",
) -> CoxResult:
    """Estimate oxide capacitance Cox from the robust mean of accumulation capacitance."""
    if capacitance_unit not in {"F", "F"}:
        raise ValueError(
            "Capacitance must use 'F'; "
            f"received '{capacitance_unit}'."
        )
    capacitance = validate_array(
        capacitance_array,
        "capacitance_array",
        minimum_points=5,
    )
    if np.any(capacitance < 0.0):
        raise ValueError("capacitance_array contains negative capacitance.")
    if not np.isfinite(top_fraction) or not 0.05 <= top_fraction <= 0.10:
        raise ValueError("top_fraction must be between 0.05 and 0.10.")

    point_count = max(2, int(np.ceil(capacitance.size * top_fraction)))
    accumulation_values = np.sort(capacitance)[-point_count:]
    filtered_values = _remove_outliers(accumulation_values)
    if filtered_values.size == 0:
        raise ValueError("No valid accumulation-region points remain.")

    cox = float(np.mean(filtered_values))
    if cox <= 0.0 or not np.isfinite(cox):
        raise ValueError("Extracted oxide capacitance is not positive and finite.")

    confidence = _calculate_confidence(filtered_values, cox)
    return CoxResult(
        cox=cox,
        confidence=confidence,
        points_used=int(filtered_values.size),
    )


def _remove_outliers(values: np.ndarray) -> np.ndarray:
    if values.size < 4:
        return values

    first_quartile, third_quartile = np.percentile(values, [25.0, 75.0])
    interquartile_range = third_quartile - first_quartile
    if interquartile_range == 0.0:
        median = float(np.median(values))
        tolerance = max(abs(median) * 1.0e-6, np.finfo(float).tiny)
        median_matches = values[np.abs(values - median) <= tolerance]
        return median_matches if median_matches.size else values

    lower_bound = first_quartile - 1.5 * interquartile_range
    upper_bound = third_quartile + 1.5 * interquartile_range
    return values[(values >= lower_bound) & (values <= upper_bound)]


def _calculate_confidence(values: np.ndarray, mean_value: float) -> float:
    if values.size == 1:
        return 1.0
    coefficient_of_variation = float(np.std(values, ddof=1) / mean_value)
    return float(np.clip(1.0 - coefficient_of_variation, 0.0, 1.0))

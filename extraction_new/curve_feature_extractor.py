"""Curve feature extraction for MOSCAP-X."""

from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

try:
    from ..extraction.phase2_validation import (
        validate_array,
    )
except ImportError:
    from extraction.phase2_validation import (
        validate_array,
    )



@dataclass(frozen=True, slots=True)
class CurveFeatures:
    """Substrate-independent mathematical features of a C-V curve."""

    voltage: np.ndarray
    capacitance: np.ndarray

    first_derivative: np.ndarray
    second_derivative: np.ndarray
    point_count: int
    average_voltage_step: float
    noise_estimate: float    
    
def extract_curve_features(
    voltage_array: Sequence[float] | np.ndarray,
    capacitance_array: Sequence[float] | np.ndarray,
) -> CurveFeatures:
    """
    Extract mathematical features from a C-V curve.

    This function performs no MOS physics.
    It only computes derivatives and descriptive quantities.
    """

    voltage = validate_array(
        voltage_array,
        "voltage_array",
        minimum_points=5,
    )

    capacitance = validate_array(
        capacitance_array,
        "capacitance_array",
        minimum_points=5,
    )

    if voltage.size != capacitance.size:
        raise ValueError(
            "voltage_array and capacitance_array must have equal length."
        )
    
    if np.any(np.diff(voltage) == 0.0):
        raise ValueError(
            "Duplicate voltage values prevent derivative calculation."
        )

    first_derivative = _compute_first_derivative(
        voltage,
        capacitance,
    )

    second_derivative = _compute_second_derivative(
        voltage,
        first_derivative,
    )

    average_voltage_step = _average_voltage_step(
        voltage,
    )

    noise_estimate = _estimate_noise(
        first_derivative,
    )

    return CurveFeatures(
        voltage=voltage,
        capacitance=capacitance,
        first_derivative=first_derivative,
        second_derivative=second_derivative,
        point_count=int(voltage.size),
        average_voltage_step=average_voltage_step,
        noise_estimate=noise_estimate,
    )

def _compute_first_derivative(
    voltage: np.ndarray,
    capacitance: np.ndarray,
) -> np.ndarray:
    """Return dC/dV."""

    return np.gradient(
        capacitance,
        voltage,
    )


def _compute_second_derivative(
    voltage: np.ndarray,
    first_derivative: np.ndarray,
) -> np.ndarray:
    """Return d²C/dV²."""

    return np.gradient(
        first_derivative,
        voltage,
    )


def _average_voltage_step(
    voltage: np.ndarray,
) -> float:
    """Return the average voltage spacing."""

    return float(
        np.mean(
            np.abs(np.diff(voltage))
        )
    )


def _estimate_noise(
    first_derivative: np.ndarray,
) -> float:
    """
    Estimate curve noise.

    Version 1 uses the standard deviation of dC/dV.
    """

    return float(
        np.std(
            first_derivative,
            ddof=1,
        )
    )    
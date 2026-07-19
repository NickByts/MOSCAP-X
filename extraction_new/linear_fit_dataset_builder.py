"""Build the linear regression dataset from the depletion region."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .curve_feature_extractor import CurveFeatures
    from .linear_region_detector import LinearRegionResult
except ImportError:
    from extraction_new.curve_feature_extractor import CurveFeatures
    from extraction_new.linear_region_detector import ( LinearRegionResult,)


@dataclass(frozen=True, slots=True)
class LinearFitDataset:
    """
    Dataset supplied to the linear regression engine.
    """

    voltage: np.ndarray

    capacitance: np.ndarray

    inverse_c2: np.ndarray

    point_count: int


def build_linear_fit_dataset(
    features: CurveFeatures,
    region: LinearRegionResult,
) -> LinearFitDataset:
    """
    Build the depletion-region dataset required for
    1/C²-V linear regression.
    """
    voltage = features.voltage[
        region.start_index : region.end_index + 1
    ]

    capacitance = features.capacitance[
        region.start_index : region.end_index + 1
    ]
    _validate_dataset(
        voltage,
        capacitance,
    )

    inverse_c2 = _calculate_inverse_c2(
        capacitance,
    )

    return LinearFitDataset(
        voltage=voltage,
        capacitance=capacitance,
        inverse_c2=inverse_c2,
        point_count=voltage.size,
    )


def _calculate_inverse_c2(
    capacitance: np.ndarray,
) -> np.ndarray:
    """
    Compute 1/C².
    """

    return 1.0 / np.square(capacitance)


def _validate_dataset(
    voltage: np.ndarray,
    capacitance: np.ndarray,
) -> None:
    """
    Validate the regression dataset.
    """

    if voltage.size != capacitance.size:
        raise ValueError(
            "Voltage and capacitance arrays must have equal length."
        )

    if voltage.size < 2:
        raise ValueError(
            "At least two depletion points are required."
        )

    if not np.all(np.isfinite(voltage)):
        raise ValueError(
            "Voltage contains non-finite values."
        )

    if not np.all(np.isfinite(capacitance)):
        raise ValueError(
            "Capacitance contains non-finite values."
        )

    if np.any(capacitance <= 0.0):
        raise ValueError(
            "Capacitance must be strictly positive."
        )
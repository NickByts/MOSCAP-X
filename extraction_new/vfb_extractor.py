"""Automatic flat-band voltage extraction for MOSCAP-X."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .curve_feature_extractor import CurveFeatures
    from .cfb_extractor import CfbResult
    from .linear_region_detector import LinearRegionResult

except ImportError:

    from extraction_new.curve_feature_extractor import CurveFeatures
    from extraction_new.cfb_extractor import CfbResult


@dataclass(frozen=True, slots=True)
class VfbResult:
    """
    Automatic flat-band voltage extraction.
    """

    vfb: float

    crossing_index: int

    interpolated: bool


def calculate_vfb(
    features: CurveFeatures,
    cfb: CfbResult,
    linear_region: LinearRegionResult,
) -> VfbResult:
    """
    Estimate the flat-band voltage from the measured C-V curve.

    Method
    ------

    1. Search for a crossing of CFB.
    2. Perform linear interpolation.
    3. If no crossing exists, use the nearest point.
    """
    search_start = 0

    search_end = linear_region.end_index

    voltage = np.asarray(
        features.voltage,
        dtype=float,
    )

    capacitance = np.asarray(
        features.capacitance,
        dtype=float,
    )

    cfb_value = float(
        cfb.cfb
    )

    _validate_inputs(
        voltage,
        capacitance,
        cfb_value,
    )


    delta = capacitance - cfb_value

    for index in range(
        search_start,
        search_end,
    ):

        if delta[index] == 0.0:

            return VfbResult(
                vfb=float(voltage[index]),
                crossing_index=index,
                interpolated=False,
            )

        if delta[index] * delta[index + 1] < 0.0:

            vfb = _linear_interpolation(
                voltage[index],
                voltage[index + 1],
                capacitance[index],
                capacitance[index + 1],
                cfb_value,
            )

            return VfbResult(
                vfb=vfb,
                crossing_index=index,
                interpolated=True,
            )

    search_slice = slice(
        search_start,
        search_end + 1,
    )

    nearest_local = int(
        np.argmin(
            np.abs(
                delta[search_slice]
            )
        )
    )

    nearest_index = (
        search_start
        + nearest_local
    )

    return VfbResult(
        vfb=float(
            voltage[nearest_index]
        ),
        crossing_index=nearest_index,
        interpolated=False,
    )


def _linear_interpolation(
    v1: float,
    v2: float,
    c1: float,
    c2: float,
    target: float,
) -> float:
    """
    Linear interpolation.

    V = V1 +
        ((Target-C1)/(C2-C1))
        (V2-V1)
    """

    if c2 == c1:
        return float(v1)

    return float(
        v1
        + (
            (target - c1)
            / (c2 - c1)
        )
        * (v2 - v1)
    )


def _validate_inputs(
    voltage: np.ndarray,
    capacitance: np.ndarray,
    cfb: float,
) -> None:
    """
    Validate extractor inputs.
    """

    if voltage.size != capacitance.size:
        raise ValueError(
            "Voltage and capacitance arrays must have equal length."
        )

    if voltage.size < 2:
        raise ValueError(
            "At least two measurement points are required."
        )

    if not np.all(np.isfinite(voltage)):
        raise ValueError(
            "Voltage contains non-finite values."
        )

    if not np.all(np.isfinite(capacitance)):
        raise ValueError(
            "Capacitance contains non-finite values."
        )

    if not np.isfinite(cfb):
        raise ValueError(
            "CFB is not finite."
        )

    if cfb <= 0.0:
        raise ValueError(
            "CFB must be greater than zero."
        )
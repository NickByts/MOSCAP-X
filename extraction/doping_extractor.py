"""Doping concentration extraction for MOSCAP-X Phase 1B."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ELEMENTARY_CHARGE_C = 1.602176634e-19
EPSILON_SI_F_PER_CM = 1.05e-12
DOPING_UNITS = "cm^-3"
N_TYPE = "N-Type"
P_TYPE = "P-Type"


@dataclass(frozen=True)
class DopingResult:
    """Extracted semiconductor doping concentration."""

    doping_value: float
    nd: float
    na: float
    substrate_type: str
    units: str = DOPING_UNITS

    @property
    def scientific_notation(self) -> str:
        value = f"{self.doping_value:.3E}".replace("E+", "E")
        return f"{value} {self.units}"


def extract_doping_concentration(
    slope: float,
    area_cm2: float,
) -> DopingResult:
    """Extract positive Nd or Na from the 1/C^2-V slope magnitude."""
    numeric_slope = _validate_slope(slope)
    numeric_area = _validate_area(area_cm2)
    slope_magnitude = abs(numeric_slope)

    denominator = (
        ELEMENTARY_CHARGE_C
        * EPSILON_SI_F_PER_CM
        * numeric_area**2
        * slope_magnitude
    )
    if denominator == 0.0 or not np.isfinite(denominator):
        raise ValueError("Doping extraction denominator is zero or non-finite.")

    doping_value = 2.0 / denominator

    if numeric_slope < 0:
        return _build_result(
            doping_value=doping_value,
            nd=doping_value,
            na=0.0,
            substrate_type=N_TYPE,
        )

    return _build_result(
        doping_value=doping_value,
        nd=0.0,
        na=doping_value,
        substrate_type=P_TYPE,
    )


def extract_doping(
    slope: float,
    area_cm2: float,
) -> DopingResult:
    """Extract doping concentration; alias for extract_doping_concentration."""
    return extract_doping_concentration(slope, area_cm2)


def _validate_slope(slope: float) -> float:
    numeric_slope = float(slope)
    if not np.isfinite(numeric_slope):
        raise ValueError("Slope must be a finite numeric value.")

    if numeric_slope == 0.0:
        raise ValueError("Zero slope causes division by zero.")

    return numeric_slope


def _validate_area(area_cm2: float) -> float:
    numeric_area = float(area_cm2)
    if not np.isfinite(numeric_area):
        raise ValueError("Device area must be a finite numeric value.")

    if numeric_area <= 0.0:
        raise ValueError("Device area must be greater than zero.")

    return numeric_area


def _build_result(
    doping_value: float,
    nd: float,
    na: float,
    substrate_type: str,
) -> DopingResult:
    if not np.isfinite(doping_value) or doping_value <= 0.0:
        raise ValueError("Extracted doping concentration is non-physical.")

    return DopingResult(
        doping_value=float(doping_value),
        nd=float(nd),
        na=float(na),
        substrate_type=substrate_type,
    )

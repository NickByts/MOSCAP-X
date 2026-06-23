"""Effective oxide-charge extraction for MOSCAP-X Phase 2."""

from __future__ import annotations

import numpy as np

try:
    from .phase2_constants import ELEMENTARY_CHARGE_C
    from .phase2_validation import (
        validate_finite,
        validate_finite_positive,
        validate_unit,
    )
except ImportError:
    from phase2_constants import ELEMENTARY_CHARGE_C
    from phase2_validation import (
        validate_finite,
        validate_finite_positive,
        validate_unit,
    )


def calculate_effective_oxide_charge(
    cox_f: float,
    vfb: float,
    phi_ms: float,
    area_cm2: float | None = None,
    *,
    capacitance_unit: str = "F",
    area_unit: str = "cm^2",
) -> float:
    """Calculate effective oxide charge density in C/cm^2.

    Cox is interpreted as F/cm^2. Supplying ``area_cm2`` enables the legacy
    total-capacitance adapter, where Cox is converted from F to F/cm^2.
    """
    if capacitance_unit not in {"F", "F"}:
        validate_unit(capacitance_unit, "F", "Oxide capacitance")
    cox = validate_finite_positive(cox_f, "Oxide capacitance")
    flatband_voltage = validate_finite(vfb, "Flat-band voltage")
    work_function_difference = validate_finite(phi_ms, "phi_ms")
    if area_cm2 is not None:
        validate_unit(area_unit, "cm^2", "Area")
        area = validate_finite_positive(area_cm2, "Device area")
        

    charge_density = (cox / area ) * (
        work_function_difference - flatband_voltage
    )
    if not np.isfinite(charge_density):
        raise ValueError("Calculated effective oxide charge is non-finite.")
    return float(charge_density)


def calculate_effective_charge_density(
    qeff_c_cm2: float,
    *,
    elementary_charge_c: float = ELEMENTARY_CHARGE_C,
    charge_unit: str = "C/cm^2",
) -> float:
    """Convert effective oxide charge into effective charge density in cm^-2."""
    validate_unit(charge_unit, "C/cm^2", "Effective oxide charge")
    qeff = validate_finite(qeff_c_cm2, "Effective oxide charge")
    elementary_charge = validate_finite_positive(
        elementary_charge_c,
        "Elementary charge",
    )
    neff = qeff / elementary_charge
    if not np.isfinite(neff):
        raise ValueError("Calculated effective charge density is non-finite.")
    return float(neff)

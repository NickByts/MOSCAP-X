"""Semiconductor and flat-band capacitance extraction for Phase 2."""

from __future__ import annotations

import numpy as np

try:
    from .phase2_validation import (
        SILICON_PERMITTIVITY_F_PER_CM,
        validate_finite_positive,
        validate_unit,
    )
except ImportError:
    from phase2_validation import (
        SILICON_PERMITTIVITY_F_PER_CM,
        validate_finite_positive,
        validate_unit,
    )


def calculate_semiconductor_capacitance(
    area_cm2: float,
    debye_length_cm: float,
    *,
    permittivity_f_per_cm: float = SILICON_PERMITTIVITY_F_PER_CM,
    area_unit: str = "cm^2",
    length_unit: str = "cm",
    permittivity_unit: str = "F/cm",
) -> float:
    """
    Cs = εs * Area / Ld
    """

    validate_unit(area_unit, "cm^2", "Area")
    validate_unit(length_unit, "cm", "Debye Length")
    validate_unit(permittivity_unit, "F/cm", "Permittivity")

    area = validate_finite_positive(
        area_cm2,
        "Device area",
    )

    debye_length = validate_finite_positive(
        debye_length_cm,
        "Debye length",
    )

    permittivity = validate_finite_positive(
        permittivity_f_per_cm,
        "Semiconductor permittivity",
    )

    capacitance = (
        permittivity
        * area
        / debye_length
    )

    if not np.isfinite(capacitance):
        raise ValueError(
            "Calculated semiconductor capacitance is non-finite."
        )

    return float(capacitance)

def calculate_flatband_capacitance(
    cox_f: float,
    semiconductor_capacitance_f: float,
    *,
    capacitance_unit: str = "F",
) -> float:
    """Calculate flat-band capacitance in F."""
    if capacitance_unit not in {"F", "F"}:
        validate_unit(capacitance_unit, "F", "Capacitance")
    cox = validate_finite_positive(cox_f, "Oxide capacitance")
    semiconductor_capacitance = validate_finite_positive(
        semiconductor_capacitance_f,
        "Semiconductor capacitance",
    )
    denominator = cox + semiconductor_capacitance
    if denominator == 0.0 or not np.isfinite(denominator):
        raise ValueError("Flat-band capacitance denominator is invalid.")

    flatband_capacitance = cox * semiconductor_capacitance / denominator
    if not np.isfinite(flatband_capacitance):
        raise ValueError("Calculated flat-band capacitance is non-finite.")
    return float(flatband_capacitance)

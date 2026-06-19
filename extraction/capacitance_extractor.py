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
    depletion_width_cm: float | None = None,
    *,
    permittivity_f_per_cm: float = SILICON_PERMITTIVITY_F_PER_CM,
    area_unit: str = "cm^2",
    length_unit: str = "cm",
    permittivity_unit: str = "F/cm",
) -> float:
    """Calculate semiconductor depletion capacitance density in F/cm^2.

    The preferred call is ``calculate_semiconductor_capacitance(width_cm)``.
    The legacy two-positional-argument call ``(area_cm2, width_cm)`` remains
    supported and returns total capacitance in Farads.
    """
    validate_unit(length_unit, "cm", "Depletion width")
    validate_unit(permittivity_unit, "F/cm", "Permittivity")
    legacy_area: float | None = None
    if depletion_width_cm is None:
        width = validate_finite_positive(area_cm2, "Depletion width")
    else:
        validate_unit(area_unit, "cm^2", "Area")
        legacy_area = validate_finite_positive(area_cm2, "Device area")
        width = validate_finite_positive(depletion_width_cm, "Depletion width")
    permittivity = validate_finite_positive(
        permittivity_f_per_cm,
        "Semiconductor permittivity",
    )

    capacitance_density = permittivity / width
    capacitance = (
        capacitance_density
        if legacy_area is None
        else capacitance_density * legacy_area
    )
    if not np.isfinite(capacitance):
        raise ValueError("Calculated semiconductor capacitance is non-finite.")
    return float(capacitance)


def calculate_flatband_capacitance(
    cox_f: float,
    semiconductor_capacitance_f: float,
    *,
    capacitance_unit: str = "F/cm^2",
) -> float:
    """Calculate areal flat-band capacitance in F/cm^2."""
    if capacitance_unit not in {"F/cm^2", "F"}:
        validate_unit(capacitance_unit, "F/cm^2", "Capacitance")
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

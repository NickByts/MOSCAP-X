"""Maximum electric-field extraction for MOSCAP-X Phase 2."""

from __future__ import annotations

import numpy as np

try:
    from .phase2_validation import (
        ELEMENTARY_CHARGE_C,
        SILICON_PERMITTIVITY_F_PER_CM,
        validate_finite_positive,
        validate_unit,
    )
except ImportError:
    from phase2_validation import (
        ELEMENTARY_CHARGE_C,
        SILICON_PERMITTIVITY_F_PER_CM,
        validate_finite_positive,
        validate_unit,
    )


def calculate_max_electric_field(
    doping_cm3: float,
    depletion_width_cm: float,
    *,
    permittivity_f_per_cm: float = SILICON_PERMITTIVITY_F_PER_CM,
    doping_unit: str = "cm^-3",
    length_unit: str = "cm",
    permittivity_unit: str = "F/cm",
) -> float:
    """Calculate the maximum depletion electric field in V/cm."""
    validate_unit(doping_unit, "cm^-3", "Doping")
    validate_unit(length_unit, "cm", "Depletion width")
    validate_unit(permittivity_unit, "F/cm", "Permittivity")
    doping = validate_finite_positive(doping_cm3, "Doping concentration")
    width = validate_finite_positive(
        depletion_width_cm,
        "Depletion width",
    )
    permittivity = validate_finite_positive(
        permittivity_f_per_cm,
        "Silicon permittivity",
    )

    electric_field = ELEMENTARY_CHARGE_C * doping * width / permittivity
    if not np.isfinite(electric_field):
        raise ValueError("Calculated electric field is non-finite.")
    return float(electric_field)

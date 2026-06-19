"""Maximum depletion-width extraction for MOSCAP-X Phase 2."""

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


def calculate_max_depletion_width(
    phi_f: float,
    doping_cm3: float,
    *,
    permittivity_f_per_cm: float = SILICON_PERMITTIVITY_F_PER_CM,
    doping_unit: str = "cm^-3",
    permittivity_unit: str = "F/cm",
) -> float:
    """Calculate maximum depletion width using surface potential 2*phi_f."""
    validate_unit(doping_unit, "cm^-3", "Doping")
    validate_unit(permittivity_unit, "F/cm", "Permittivity")
    fermi_potential = validate_finite_positive(phi_f, "Fermi potential")
    doping = validate_finite_positive(doping_cm3, "Doping concentration")
    permittivity = validate_finite_positive(
        permittivity_f_per_cm,
        "Silicon permittivity",
    )

    surface_potential = 2.0 * fermi_potential
    radicand = (
        2.0
        * permittivity
        * surface_potential
        / (ELEMENTARY_CHARGE_C * doping)
    )
    if radicand <= 0.0 or not np.isfinite(radicand):
        raise ValueError("Depletion-width radicand must be positive and finite.")
    return float(np.sqrt(radicand))

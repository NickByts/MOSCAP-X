"""Debye length extraction for MOSCAP-X Phase 2."""

from __future__ import annotations

import numpy as np

try:
    from .phase2_validation import (
        BOLTZMANN_CONSTANT_J_PER_K,
        ELEMENTARY_CHARGE_C,
        SILICON_PERMITTIVITY_F_PER_CM,
        validate_finite_positive,
        validate_substrate_type,
        validate_unit,
    )
except ImportError:
    from phase2_validation import (
        BOLTZMANN_CONSTANT_J_PER_K,
        ELEMENTARY_CHARGE_C,
        SILICON_PERMITTIVITY_F_PER_CM,
        validate_finite_positive,
        validate_substrate_type,
        validate_unit,
    )


def calculate_debye_length(
    doping_cm3: float,
    temperature_k: float,
    substrate_type: str,
    *,
    permittivity_f_per_cm: float = SILICON_PERMITTIVITY_F_PER_CM,
    doping_unit: str = "cm^-3",
    temperature_unit: str = "K",
    permittivity_unit: str = "F/cm",
) -> float:
    """Calculate the semiconductor Debye length in centimeters."""
    validate_unit(doping_unit, "cm^-3", "Doping")
    validate_unit(temperature_unit, "K", "Temperature")
    validate_unit(permittivity_unit, "F/cm", "Permittivity")
    doping = validate_finite_positive(doping_cm3, "Doping concentration")
    temperature = validate_finite_positive(temperature_k, "Temperature")
    permittivity = validate_finite_positive(
        permittivity_f_per_cm,
        "Silicon permittivity",
    )
    validate_substrate_type(substrate_type)

    radicand = (
        permittivity
        * BOLTZMANN_CONSTANT_J_PER_K
        * temperature
        / (ELEMENTARY_CHARGE_C**2 * doping)
    )
    if radicand <= 0.0 or not np.isfinite(radicand):
        raise ValueError("Debye-length radicand must be positive and finite.")
    return float(np.sqrt(radicand))

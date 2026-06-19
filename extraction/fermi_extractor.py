"""Fermi potential extraction for MOSCAP-X Phase 2."""

from __future__ import annotations

import numpy as np

try:
    from .phase2_validation import (
        BOLTZMANN_CONSTANT_J_PER_K,
        ELEMENTARY_CHARGE_C,
        validate_finite_positive,
        validate_unit,
    )
except ImportError:
    from phase2_validation import (
        BOLTZMANN_CONSTANT_J_PER_K,
        ELEMENTARY_CHARGE_C,
        validate_finite_positive,
        validate_unit,
    )


def calculate_fermi_potential(
    doping_cm3: float,
    intrinsic_concentration_cm3: float = 9.65e9,
    temperature_k: float = 300.0,
    *,
    doping_unit: str = "cm^-3",
    temperature_unit: str = "K",
) -> float:
    """Calculate the bulk Fermi-potential magnitude in volts."""
    validate_unit(doping_unit, "cm^-3", "Doping")
    validate_unit(temperature_unit, "K", "Temperature")
    doping = validate_finite_positive(doping_cm3, "Doping concentration")
    intrinsic = validate_finite_positive(
        intrinsic_concentration_cm3,
        "Intrinsic carrier concentration",
    )
    temperature = validate_finite_positive(temperature_k, "Temperature")
    logarithm_argument = doping / intrinsic
    if logarithm_argument <= 0.0 or not np.isfinite(logarithm_argument):
        raise ValueError("Fermi-potential logarithm argument must be positive.")

    phi_f = (
        BOLTZMANN_CONSTANT_J_PER_K
        * temperature
        / ELEMENTARY_CHARGE_C
        * np.log(logarithm_argument)
    )
    if not np.isfinite(phi_f):
        raise ValueError("Calculated Fermi potential is non-finite.")
    return float(phi_f)

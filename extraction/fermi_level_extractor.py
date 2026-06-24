"""Fermi-level extraction for MOSCAP-X Phase 2."""

from __future__ import annotations

import numpy as np

try:
    from .phase2_constants import (
        BOLTZMANN_CONSTANT_J_PER_K,
        ELEMENTARY_CHARGE_C,
    )
    from .phase2_validation import (
        validate_finite_positive,
        validate_unit,
    )
except ImportError:
    from phase2_constants import (
        BOLTZMANN_CONSTANT_J_PER_K,
        ELEMENTARY_CHARGE_C,
    )
    from phase2_validation import (
        validate_finite_positive,
        validate_unit,
    )


def calculate_fermi_level(
    doping_cm3: float,
    temperature_k: float,
    *,
    effective_density_of_states_cm3: float = 2.82e19,
    doping_unit: str = "cm^-3",
    temperature_unit: str = "K",
) -> float:
    """
    EF = (kT/q) * ln(Nc / Nd)
    """

    validate_unit(doping_unit, "cm^-3", "Doping")
    validate_unit(temperature_unit, "K", "Temperature")

    doping = validate_finite_positive(
        doping_cm3,
        "Doping concentration",
    )

    temperature = validate_finite_positive(
        temperature_k,
        "Temperature",
    )

    nc = validate_finite_positive(
        effective_density_of_states_cm3,
        "Effective density of states",
    )

    thermal_voltage = (
        BOLTZMANN_CONSTANT_J_PER_K
        * temperature
        / ELEMENTARY_CHARGE_C
    )

    ef = thermal_voltage * np.log(
        nc / doping
    )

    if not np.isfinite(ef):
        raise ValueError(
            "Calculated Fermi level is non-finite."
        )

    return float(ef)
"""Material-property calculations for MOSCAP-X Phase 2."""

from __future__ import annotations

import numpy as np

try:
    from .phase2_constants import VACUUM_PERMITTIVITY_F_PER_CM
    from .phase2_validation import validate_finite_positive, validate_unit
except ImportError:
    from phase2_constants import VACUUM_PERMITTIVITY_F_PER_CM
    from phase2_validation import validate_finite_positive, validate_unit


def calculate_semiconductor_permittivity(
    relative_permittivity: float,
    *,
    vacuum_permittivity_f_per_cm: float = VACUUM_PERMITTIVITY_F_PER_CM,
    permittivity_unit: str = "F/cm",
) -> float:
    """Calculate absolute semiconductor permittivity in F/cm."""
    validate_unit(permittivity_unit, "F/cm", "Permittivity")
    epsilon_r = validate_finite_positive(
        relative_permittivity,
        "Relative permittivity",
    )
    epsilon_0 = validate_finite_positive(
        vacuum_permittivity_f_per_cm,
        "Vacuum permittivity",
    )

    permittivity = epsilon_r * epsilon_0
    if not np.isfinite(permittivity):
        raise ValueError("Calculated semiconductor permittivity is non-finite.")
    return float(permittivity)

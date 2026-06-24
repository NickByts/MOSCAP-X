"""Image-force barrier lowering extraction for MOSCAP-X Phase 2."""

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


def calculate_image_force_barrier_lowering(
    electric_field_v_cm: float,
    permittivity_f_per_cm: float,
    *,
    field_unit: str = "V/cm",
    permittivity_unit: str = "F/cm",
) -> float:
    """
    Δφb = sqrt(q * |Em| / (4π εs))
    """

    validate_unit(field_unit, "V/cm", "Electric field")
    validate_unit(permittivity_unit, "F/cm", "Permittivity")

    electric_field = abs(
        validate_finite(
            electric_field_v_cm,
            "Electric field",
        )
    )

    permittivity = validate_finite_positive(
        permittivity_f_per_cm,
        "Semiconductor permittivity",
    )

    delta_phi_b = np.sqrt(
        (
            ELEMENTARY_CHARGE_C
            * electric_field
        )
        /
        (
            4.0
            * np.pi
            * permittivity
        )
    )

    if not np.isfinite(delta_phi_b):
        raise ValueError(
            "Calculated image-force barrier lowering is non-finite."
        )

    return float(delta_phi_b)
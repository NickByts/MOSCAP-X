"""Maximum electric-field extraction for MOSCAP-X Phase 2."""

from __future__ import annotations


import numpy as np
from .phase2_validation import validate_finite

try:
    from .phase2_validation import (
       
        validate_finite_positive,
        
    )
except ImportError:
    from phase2_validation import (
        
        validate_finite_positive,
        
    )


def calculate_junction_electric_field(
    v0_v: float,
    depletion_width_cm: float,
    *,
    length_unit: str = "cm",
) -> float:
    """Calculate the junction electric field in V/cm."""
    
    v0 = validate_finite(
        v0_v,
        "Intercept voltage V0",
    )
    width = validate_finite_positive(
        depletion_width_cm,
        "Depletion width",
    )
    electric_field = (
        2.0
        * v0
        / width
    )
    if not np.isfinite(electric_field):
        raise ValueError("Calculated electric field is non-finite.")
    return float(electric_field)

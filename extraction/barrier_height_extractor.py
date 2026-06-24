"""Barrier-height extraction for MOSCAP-X Phase 2."""

from __future__ import annotations

import numpy as np

try:
    from .phase2_validation import validate_finite
except ImportError:
    from phase2_validation import validate_finite


def calculate_barrier_height(
    vd_v: float,
    ef_v: float,
    delta_phi_b_v: float,
) -> float:
    """
    φb = Vd + EF − Δφb
    """

    vd = validate_finite(
        vd_v,
        "Diffusion potential",
    )

    ef = validate_finite(
        ef_v,
        "Fermi level",
    )

    delta_phi_b = validate_finite(
        delta_phi_b_v,
        "Image-force barrier lowering",
    )

    barrier_height = (
        vd
        + ef
        - delta_phi_b
    )

    if not np.isfinite(barrier_height):
        raise ValueError(
            "Calculated barrier height is non-finite."
        )

    return float(barrier_height)
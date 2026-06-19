"""Semiconductor substrate type detection for MOSCAP-X Phase 1B."""

from __future__ import annotations

from typing import Literal

import numpy as np

SubstrateType = Literal["N-Type", "P-Type"]


def detect_substrate_type(slope: float) -> SubstrateType:
    """Detect semiconductor substrate type using only the slope sign."""
    numeric_slope = float(slope)

    if not np.isfinite(numeric_slope):
        raise ValueError("Slope must be a finite numeric value.")

    if numeric_slope < 0.0:
        return "N-Type"

    if numeric_slope > 0.0:
        return "P-Type"

    raise ValueError("Zero slope cannot determine substrate type.")

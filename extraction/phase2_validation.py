"""Shared validation and constants for MOSCAP-X Phase 2 extraction."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

try:
    from .phase2_constants import (
        BOLTZMANN_CONSTANT_J_PER_K,
        ELEMENTARY_CHARGE_C,
        VACUUM_PERMITTIVITY_F_PER_CM,
    )
except ImportError:
    from phase2_constants import (
        BOLTZMANN_CONSTANT_J_PER_K,
        ELEMENTARY_CHARGE_C,
        VACUUM_PERMITTIVITY_F_PER_CM,
    )

# Compatibility export for existing extractor defaults. New orchestration derives
# material permittivity from relative permittivity and vacuum permittivity.
SILICON_RELATIVE_PERMITTIVITY = 11.7
SILICON_PERMITTIVITY_F_PER_CM = (
    SILICON_RELATIVE_PERMITTIVITY * VACUUM_PERMITTIVITY_F_PER_CM
)


def validate_finite_positive(value: float, name: str) -> float:
    """Return a finite positive float or raise a descriptive exception."""
    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite.")
    if numeric_value <= 0.0:
        raise ValueError(f"{name} must be greater than zero.")
    return numeric_value


def validate_finite(value: float, name: str) -> float:
    """Return a finite float or raise a descriptive exception."""
    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite.")
    return numeric_value


def validate_array(
    values: Sequence[float] | np.ndarray,
    name: str,
    minimum_points: int = 1,
) -> np.ndarray:
    """Validate and return a one-dimensional finite numeric array."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size < minimum_points:
        raise ValueError(
            f"{name} requires at least {minimum_points} data points."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def validate_unit(actual: str, expected: str, quantity: str) -> None:
    """Require an explicit unit string to match the Phase 2 CGS unit."""
    if actual.strip() != expected:
        raise ValueError(
            f"{quantity} must use '{expected}' units; received '{actual}'."
        )


def validate_substrate_type(substrate_type: str) -> str:
    """Validate and normalize the Phase 1B substrate type output."""
    normalized = substrate_type.strip()
    if normalized not in {"N-Type", "P-Type"}:
        raise ValueError("substrate_type must be 'N-Type' or 'P-Type'.")
    return normalized

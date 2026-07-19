"""Automatic V-intercept extraction for MOSCAP-X."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from ..fitting.linear_regression import FitResult
except ImportError:
    from fitting.linear_regression import FitResult


@dataclass(frozen=True, slots=True)
class VInterceptResult:
    """
    Result of automatic V-intercept extraction.
    """

    vintercept: float


def calculate_vintercept(
    fit: FitResult,
) -> VInterceptResult:
    """
    Calculate the voltage-axis intercept of the
    depletion-region linear regression.

    The voltage intercept is

        V0 = -Intercept / Slope

    Returns
    -------
    VInterceptResult
    """

    slope = float(fit.slope)
    intercept = float(fit.intercept)

    if not np.isfinite(slope):
        raise ValueError(
            "Regression slope is not finite."
        )

    if not np.isfinite(intercept):
        raise ValueError(
            "Regression intercept is not finite."
        )

    if slope == 0.0:
        raise ValueError(
            "Cannot calculate V-intercept because the regression slope is zero."
        )

    vintercept = -intercept / slope

    if not np.isfinite(vintercept):
        raise ValueError(
            "Calculated V-intercept is not finite."
        )

    return VInterceptResult(
        vintercept=float(vintercept),
    )
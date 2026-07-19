"""Automatic semiconductor flat-band capacitance extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .debye_length_extractor import (
        DebyeLengthResult,
    )

    from .measurement_context import (
        MeasurementContext,
    )

except ImportError:

    from extraction_new.debye_length_extractor import (
        DebyeLengthResult,
    )

    from extraction_new.measurement_context import (
        MeasurementContext,
    )


EPSILON_0 = 8.854187817e-14  # F/cm


@dataclass(frozen=True, slots=True)
class CsfbResult:
    """
    Semiconductor flat-band capacitance.
    """

    csfb: float


def calculate_csfb(
    debye: DebyeLengthResult,
    context: MeasurementContext,
    area_cm2: float,
) -> CsfbResult:
    """
    Calculate semiconductor flat-band capacitance.

    Formula
    -------

        CsFB = εs A / LD

    where

        εs = εr ε0
    """

    epsilon_r = float(
        context.relative_permittivity
    )

    ld = float(
        debye.debye_length
    )

    area = float(
        area_cm2
    )

    _validate_inputs(
        epsilon_r,
        ld,
        area,
    )

    epsilon_s = (
        epsilon_r
        * EPSILON_0
    )

    csfb = (
        epsilon_s
        * area
        / ld
    )

    if (
        not np.isfinite(csfb)
        or csfb <= 0.0
    ):
        raise ValueError(
            "Calculated CsFB is non-physical."
        )

    return CsfbResult(
        csfb=float(csfb),
    )


def _validate_inputs(
    epsilon_r: float,
    ld: float,
    area: float,
) -> None:

    if epsilon_r <= 0.0:
        raise ValueError(
            "Relative permittivity must be greater than zero."
        )

    if ld <= 0.0:
        raise ValueError(
            "Debye length must be greater than zero."
        )

    if area <= 0.0:
        raise ValueError(
            "Device area must be greater than zero."
        )
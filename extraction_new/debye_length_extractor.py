"""Automatic Debye length extraction for MOSCAP-X."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .measurement_context import MeasurementContext
    from ..extraction.doping_extractor import DopingResult
except ImportError:
    from extraction_new.measurement_context import MeasurementContext
    from extraction.doping_extractor import DopingResult


# ---------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------

EPSILON_0 = 8.854187817e-14      # F/cm
BOLTZMANN = 1.380649e-23         # J/K
ELEMENTARY_CHARGE = 1.602176634e-19  # C


# ---------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DebyeLengthResult:
    """
    Automatic Debye length extraction result.
    """

    debye_length: float


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def calculate_debye_length(
    context: MeasurementContext,
    doping: DopingResult,
) -> DebyeLengthResult:
    """
    Calculate the Debye length.

    Formula
    -------

              εs kT
    LD = √ ------------
            q² N

    where

        εs = εr ε0
    """

    relative_permittivity = float(
        context.relative_permittivity
    )

    temperature = float(
        context.temperature_k
    )

    concentration = float(
        doping.doping_value
    )

    _validate_inputs(
        relative_permittivity,
        temperature,
        concentration,
    )

    epsilon_s = (
        relative_permittivity
        * EPSILON_0
    )

    numerator = (
        epsilon_s
        * BOLTZMANN
        * temperature
    )

    denominator = (
        ELEMENTARY_CHARGE**2
        * concentration
    )

    debye_length = np.sqrt(
        numerator / denominator
    )

    if (
        not np.isfinite(debye_length)
        or debye_length <= 0.0
    ):
        raise ValueError(
            "Calculated Debye length is non-physical."
        )

    return DebyeLengthResult(
        debye_length=float(debye_length),
    )


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def _validate_inputs(
    relative_permittivity: float,
    temperature: float,
    concentration: float,
) -> None:

    if relative_permittivity <= 0.0:
        raise ValueError(
            "Relative permittivity must be greater than zero."
        )

    if temperature <= 0.0:
        raise ValueError(
            "Temperature must be greater than zero."
        )

    if concentration <= 0.0:
        raise ValueError(
            "Doping concentration must be greater than zero."
        )
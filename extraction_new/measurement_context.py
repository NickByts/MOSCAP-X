"""Measurement context for MOSCAP-X.

This module converts user inputs into a reusable physics description.

It performs no curve analysis and contains no extraction logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


MaterialType = Literal[
    "Silicon",
    "Germanium",
]

SubstrateType = Literal[
    "P-Type",
    "N-Type",
]

MeasurementMode = Literal[
    "High Frequency",
    "Low Frequency",
]

Side = Literal[
    "Left",
    "Right",
]


@dataclass(frozen=True, slots=True)
class MeasurementContext:
    """
    Physics description of the MOS capacitor measurement.

    This object contains only information already known before
    any analysis of the measured C-V curve begins.
    """

    material: MaterialType

    substrate_type: SubstrateType

    frequency_hz: float

    temperature_k: float

    measurement_mode: MeasurementMode

    expected_accumulation_side: Side

    expected_inversion_side: Side

    expected_depletion_location: str

    relative_permittivity: float

    intrinsic_carrier_concentration_cm3: float


def build_measurement_context(
    *,
    material: MaterialType,
    substrate_type: SubstrateType,
    temperature_k: float,
) -> MeasurementContext:
    """
    Build the reusable measurement context.

    No measured data are analysed here.
    """

    

    if temperature_k <= 0.0:
        raise ValueError(
            "temperature_k must be greater than zero."
        )

    frequency_hz = 1.0e6

    measurement_mode: MeasurementMode = "High Frequency"

    (
        accumulation_side,
        inversion_side,
    ) = _expected_sides(
        substrate_type,
    )

    (
        relative_permittivity,
        intrinsic_concentration,
    ) = _material_properties(
        material,
        temperature_k,
    )

    return MeasurementContext(
        material=material,
        substrate_type=substrate_type,
        frequency_hz=frequency_hz,
        temperature_k=temperature_k,
        measurement_mode=measurement_mode,
        expected_accumulation_side=accumulation_side,
        expected_inversion_side=inversion_side,
        expected_depletion_location="Middle",
        relative_permittivity=relative_permittivity,
        intrinsic_carrier_concentration_cm3=intrinsic_concentration,
    )




def _expected_sides(
    substrate_type: SubstrateType,
) -> tuple[Side, Side]:
    """
    Return expected accumulation
    and inversion sides.
    """

    if substrate_type == "P-Type":
        return (
            "Left",
            "Right",
        )

    return (
        "Right",
        "Left",
    )


def _material_properties(
    material: MaterialType,
    temperature_k: float,
) -> tuple[float, float]:
    """
    Return material properties.

    Current Version:
    constants at approximately 300 K.
    """

    if material == "Silicon":

        return (
            11.7,
            9.65e9,
        )

    if material == "Germanium":

        return (
            16.0,
            2.4e13,
        )

    raise ValueError(
        f"Unsupported material '{material}'."
    )
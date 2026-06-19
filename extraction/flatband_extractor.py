"""Flat-band voltage extraction for MOSCAP-X Phase 2."""

from __future__ import annotations

from dataclasses import dataclass

try:
    from .phase2_validation import (
        validate_finite,
        validate_finite_positive,
        validate_substrate_type,
        validate_unit,
    )
except ImportError:
    from phase2_validation import (
        validate_finite,
        validate_finite_positive,
        validate_substrate_type,
        validate_unit,
    )


@dataclass(frozen=True)
class FlatbandResult:
    """Metal-semiconductor work-function difference and flat-band voltage."""

    phi_ms: float
    vfb: float


def calculate_flatband_voltage(
    metal_work_function_ev: float,
    electron_affinity_ev: float,
    bandgap_ev: float,
    phi_f: float,
    substrate_type: str,
    *,
    energy_unit: str = "eV",
) -> FlatbandResult:
    """Calculate ideal phi_ms and Vfb from standard MOS work functions."""
    validate_unit(energy_unit, "eV", "Work-function energy")
    metal_work_function = validate_finite(
        metal_work_function_ev,
        "Metal work function",
    )
    electron_affinity = validate_finite_positive(
        electron_affinity_ev,
        "Electron affinity",
    )
    bandgap = validate_finite_positive(bandgap_ev, "Bandgap")
    fermi_potential = validate_finite(phi_f, "Fermi potential")
    if fermi_potential < 0.0:
        raise ValueError("Fermi potential magnitude cannot be negative.")
    normalized_type = validate_substrate_type(substrate_type)

    intrinsic_work_function = electron_affinity + bandgap / 2.0
    if normalized_type == "P-Type":
        semiconductor_work_function = intrinsic_work_function + fermi_potential
    else:
        semiconductor_work_function = intrinsic_work_function - fermi_potential

    phi_ms = metal_work_function - semiconductor_work_function
    return FlatbandResult(phi_ms=float(phi_ms), vfb=float(phi_ms))

"""Typed inputs and material properties for MOSCAP-X Phase 2."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Phase2Inputs:
    """User and upstream inputs using the Phase 2 areal unit convention."""

    area_cm2: float
    temperature_k: float

    doping_cm3: float
    substrate_type: str

    cox_f_cm2: float
    vfb_v: float

    phi_m_ev: float


@dataclass(frozen=True, slots=True)
class Phase2MaterialProperties:
    """Semiconductor properties kept separate from user/device inputs."""

    intrinsic_concentration_cm3: float
    bandgap_ev: float
    electron_affinity_ev: float
    relative_permittivity: float

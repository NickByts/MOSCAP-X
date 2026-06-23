"""Typed output container for MOSCAP-X Phase 2."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Phase2Results:
    """Complete Phase 2 result using consistent CGS and areal units."""

    cox_f: float

    vfb_v: float

    vd_v: float

    phi_f_v: float

    ld_cm: float

    wd_cm: float

    em_v_cm: float

    cs_f: float

    cfb_f: float

    phi_ms_v: float

    qeff_c_cm2: float

    neff_cm2: float

    @property
    def phi_f(self) -> float:
        return self.phi_f_v

    @property
    def ld(self) -> float:
        return self.ld_cm

    @property
    def wd(self) -> float:
        return self.wd_cm

    @property
    def em(self) -> float:
        return self.em_v_cm

    @property
    def cs(self) -> float:
        return self.cs_f

    @property
    def cfb(self) -> float:
        return self.cfb_f

    @property
    def phi_ms(self) -> float:
        return self.phi_ms_v

    @property
    def qeff(self) -> float:
        return self.qeff_c_cm2

    @property
    def neff(self) -> float:
        return self.neff_cm2

    @property
    def cox(self) -> float:
        return self.cox_f


    @property
    def vfb(self) -> float:
        return self.vfb_v
    
    @property
    def vd(self) -> float:
        return self.vd_v
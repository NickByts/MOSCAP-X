"""Pure Phase 2 orchestration for MOSCAP-X."""

from __future__ import annotations

from .capacitance_extractor import (
    calculate_flatband_capacitance,
    calculate_semiconductor_capacitance,
)
from .debye_extractor import calculate_debye_length
from .depletion_extractor import calculate_depletion_width
from .electric_field_extractor import calculate_junction_electric_field
from .fermi_extractor import calculate_fermi_potential
from .flatband_extractor import calculate_flatband_voltage
from .material_extractor import calculate_semiconductor_permittivity
from .oxide_charge_extractor import (
    calculate_effective_charge_density,
    calculate_effective_oxide_charge,
)

from .barrier_lowering_extractor import (
    calculate_image_force_barrier_lowering,
)

from .phase2_constants import (
    BOLTZMANN_CONSTANT_J_PER_K,
    ELEMENTARY_CHARGE_C,
)
from .fermi_level_extractor import (
    calculate_fermi_level,
)
from .barrier_height_extractor import (
    calculate_barrier_height,
)

from .phase2_models import Phase2Inputs, Phase2MaterialProperties
from .phase2_results import Phase2Results
from .phase2_validation import (
    validate_finite,
    validate_finite_positive,
    validate_substrate_type,
)


def calculate_phase2_summary(
    inputs: Phase2Inputs,
    materials: Phase2MaterialProperties,
) -> Phase2Results:
    """Run Phase 2 extractors in dependency order."""
    _validate_phase2_inputs(inputs, materials)

    permittivity_f_per_cm = calculate_semiconductor_permittivity(
        materials.relative_permittivity,
    )
    phi_f_v = calculate_fermi_potential(
        inputs.doping_cm3,
        materials.intrinsic_concentration_cm3,
        inputs.temperature_k,
    )

    thermal_voltage_v = (
        BOLTZMANN_CONSTANT_J_PER_K
        * inputs.temperature_k
        / ELEMENTARY_CHARGE_C
    )

    vd_v = inputs.v0_v + thermal_voltage_v

    ld_cm = calculate_debye_length(
        inputs.doping_cm3,
        inputs.temperature_k,
        inputs.substrate_type,
        permittivity_f_per_cm=permittivity_f_per_cm,
    )
    wd_cm = calculate_depletion_width(
        vd_v,
        inputs.doping_cm3,
        permittivity_f_per_cm=permittivity_f_per_cm,
    )


    em_v_cm = calculate_junction_electric_field(
        inputs.v0_v,
        wd_cm,
    )

    delta_phi_b_v = calculate_image_force_barrier_lowering(
        em_v_cm,
        permittivity_f_per_cm,
    )

    ef_v = calculate_fermi_level(
        inputs.doping_cm3,
        inputs.temperature_k,
    )

    phi_b_v = calculate_barrier_height(
        vd_v,
        ef_v,
        delta_phi_b_v,
    )

    cs_f = calculate_semiconductor_capacitance(
        inputs.area_cm2,
        ld_cm,
        permittivity_f_per_cm=permittivity_f_per_cm,
    )
    cfb_f = calculate_flatband_capacitance(
        inputs.cox_f,
        cs_f,
    )
    flatband = calculate_flatband_voltage(
        inputs.phi_m_ev,
        materials.electron_affinity_ev,
        materials.bandgap_ev,
        phi_f_v,
        inputs.substrate_type,
    )
    qeff_c_cm2 = calculate_effective_oxide_charge(
        inputs.cox_f,
        inputs.vfb_v,
        flatband.phi_ms,
        inputs.area_cm2,
    )
    neff_cm2 = calculate_effective_charge_density(qeff_c_cm2)

    return Phase2Results(
        cox_f=inputs.cox_f,
        vfb_v=inputs.vfb_v,
        phi_f_v=phi_f_v,
        ld_cm=ld_cm,
        wd_cm=wd_cm,
        em_v_cm=em_v_cm,
        delta_phi_b_v=delta_phi_b_v,
        ef_v=ef_v,
        cs_f=cs_f,
        cfb_f=cfb_f,
        vd_v=vd_v,
        phi_ms_v=flatband.phi_ms,
        phi_b_v=phi_b_v,
        qeff_c_cm2=qeff_c_cm2,
        neff_cm2=neff_cm2,
        
    )   


def calculate_phase2_results(
    inputs: Phase2Inputs,
    materials: Phase2MaterialProperties,
) -> Phase2Results:
    """Compatibility alias for the typed Phase 2 orchestrator."""
    return calculate_phase2_summary(inputs, materials)


def _validate_phase2_inputs(
    inputs: Phase2Inputs,
    materials: Phase2MaterialProperties,
) -> None:
    validate_finite_positive(inputs.area_cm2, "Device area")
    validate_finite_positive(inputs.temperature_k, "Temperature")
    validate_finite_positive(inputs.doping_cm3, "Doping concentration")
    validate_substrate_type(inputs.substrate_type)
    validate_finite_positive(inputs.cox_f, "Oxide capacitance")
    validate_finite(inputs.vfb_v, "Flat-band voltage")
    validate_finite(inputs.phi_m_ev, "Metal work function")

    validate_finite_positive(
        materials.intrinsic_concentration_cm3,
        "Intrinsic concentration",
    )
    validate_finite_positive(materials.bandgap_ev, "Bandgap")
    validate_finite_positive(
        materials.electron_affinity_ev,
        "Electron affinity",
    )
    validate_finite_positive(
        materials.relative_permittivity,
        "Relative permittivity",
    )


Phase2Summary = Phase2Results

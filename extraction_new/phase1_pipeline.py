"""Phase 1 pipeline for the new MOSCAP-X architecture."""

from __future__ import annotations

from dataclasses import dataclass


try:
    from .measurement_context import (
        MeasurementContext,
    )

    from .curve_feature_extractor import (
        CurveFeatures,
        extract_curve_features,
    )

    from .plateau_detector import (
        DirectedPlateauResult,
        detect_accumulation_plateau,
    )

    from .cox_extractor import (
        CoxResult,
        calculate_cox,
    )

    from .linear_region_detector import (
        LinearRegionResult,
        detect_linear_region,
    )

    from .linear_fit_dataset_builder import (
        LinearFitDataset,
        build_linear_fit_dataset,
    )

    from ..fitting.linear_regression import (
        FitResult,
        perform_linear_fit,
    )

    from ..extraction.doping_extractor import (
        DopingResult,
        extract_doping_concentration,
    )

    from .v_intercept_extractor import (
        VInterceptResult,
        calculate_vintercept,
    )

    from .debye_length_extractor import (
        DebyeLengthResult,
        calculate_debye_length,
    )

    from .csfb_extractor import (
        CsfbResult,
        calculate_csfb,
    )

    from .cfb_extractor import (
        CfbResult,
        calculate_cfb,
    )

    from .vfb_extractor import (
        VfbResult,
        calculate_vfb,
    )

except ImportError:

    from extraction_new.measurement_context import (
        MeasurementContext,
    )

    from extraction_new.curve_feature_extractor import (
        CurveFeatures,
        extract_curve_features,
    )

    from extraction_new.plateau_detector import (
        DirectedPlateauResult,
        detect_accumulation_plateau,
    )

    from extraction_new.cox_extractor import (
        CoxResult,
        calculate_cox,
    )

    from extraction_new.linear_region_detector import (
        LinearRegionResult,
        detect_linear_region,
    )

    from extraction_new.linear_fit_dataset_builder import (
        LinearFitDataset,
        build_linear_fit_dataset,
    )

    from fitting.linear_regression import (
        FitResult,
        perform_linear_fit,
    )

    from extraction.doping_extractor import (
        DopingResult,
        extract_doping_concentration,
    )

    from extraction_new.v_intercept_extractor import (
        VInterceptResult,
        calculate_vintercept,
    )

    from extraction_new.debye_length_extractor import (
        DebyeLengthResult,
        calculate_debye_length,
    )

    from extraction_new.csfb_extractor import (
        CsfbResult,
        calculate_csfb,
    )

    from extraction_new.cfb_extractor import (
        CfbResult,
        calculate_cfb,
    )

    from extraction_new.vfb_extractor import (
        VfbResult,
        calculate_vfb,
    )


@dataclass(frozen=True, slots=True)
class Phase1PipelineResult:
    """
    Complete output of the Phase 1 extraction pipeline.
    """

    context: MeasurementContext

    features: CurveFeatures

    plateau: DirectedPlateauResult

    cox: CoxResult

    linear_region: LinearRegionResult

    dataset: LinearFitDataset

    fit: FitResult

    vintercept: VInterceptResult

    doping: DopingResult

    debye: DebyeLengthResult

    csfb: CsfbResult

    cfb: CfbResult

    vfb: VfbResult


def run_phase1_pipeline(
    *,
    voltage,
    capacitance,
    context: MeasurementContext,
    area_cm2: float,
) -> Phase1PipelineResult:
    """
    Execute the complete Phase 1 extraction pipeline.

    Pipeline
    --------
    Measurement Context
            ↓
    Curve Feature Extractor
            ↓
    Directed Plateau Detector
            ↓
    Cox Extractor
            ↓
    Linear Region Detector
            ↓
    Linear Fit Dataset Builder
            ↓
    Linear Regression
            ↓
    Doping Extractor
            ↓
    Debye → CsFB → CFB → VFB
    """

    area_cm2 = _validate_area(
        area_cm2,
    )

    features = extract_curve_features(
        voltage_array=voltage,
        capacitance_array=capacitance,
    )

    plateau = detect_accumulation_plateau(
        features=features,
        context=context,
    )

    cox = calculate_cox(
        result=plateau,
    )

    linear_region = detect_linear_region(
        features=features,
        plateau_result=plateau,
    )

    dataset = build_linear_fit_dataset(
        features=features,
        region=linear_region,
    )

    fit = perform_linear_fit(
        selected_voltage=dataset.voltage,
        selected_inverse_c2=dataset.inverse_c2,
    )

    vintercept = calculate_vintercept(
        fit,
    )

    doping = extract_doping_concentration(
        slope=fit.slope,
        area_cm2=area_cm2,
    )

    debye = calculate_debye_length(
        context=context,
        doping=doping,
    )

    csfb = calculate_csfb(
        debye=debye,
        context=context,
        area_cm2=area_cm2,
    )

    cfb = calculate_cfb(
        cox=cox,
        csfb=csfb,
    )

    vfb = calculate_vfb(
        features=features,
        cfb=cfb,
        linear_region=linear_region,
    )

    return Phase1PipelineResult(
        context=context,
        features=features,
        plateau=plateau,
        cox=cox,
        linear_region=linear_region,
        dataset=dataset,
        fit=fit,
        vintercept=vintercept,
        doping=doping,
        debye=debye,
        csfb=csfb,
        cfb=cfb,
        vfb=vfb,
    )
    

def _validate_area(
    area_cm2: float,
) -> float:
    """
    Validate the device area.
    """

    numeric_area = float(area_cm2)

    if numeric_area <= 0.0:
        raise ValueError(
            "Device area must be greater than zero."
        )

    return numeric_area
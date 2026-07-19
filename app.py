"""Streamlit application entry point for MOSCAP-X Phase 1A."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

try:
    from .constants import (
        APP_DESCRIPTION,
        APP_NAME,
        CAPACITANCE_UNITS,
        DEFAULT_DEVICE_AREA_CM2,
        MIN_DEVICE_AREA_CM2,
        SAMPLE_DATA_RELATIVE_PATH,
        STANDARD_CAPACITANCE_COLUMN,
        STANDARD_VOLTAGE_COLUMN,
        SUPPORTED_UPLOAD_TYPES,
    )
    from .data_loader import load_csv, load_data
    
    from .extraction.phase2_summary import (
        Phase2Summary,
        calculate_phase2_summary,
    )
    from extraction.phase2_models import (
        Phase2Inputs,
        Phase2MaterialProperties,
    )
    from .plot_phase1_pipeline import (
        plot_phase1_cv_regions,
        plot_phase1_linear_fit,
    )
    from .plotting import plot_cv, plot_inverse_c2, plot_normalized_cv
    from .preprocessing import (
        clean_data,
        convert_capacitance_units,
        validate_data,
    )
    from .utils import (
        add_normalized_capacitance,
        dataframe_to_csv_bytes,
        figure_to_png_bytes,
        format_scientific,
    )

    from .extraction_new.measurement_context import (
        MeasurementContext,
        build_measurement_context,
    )
except ImportError:
    from constants import (
        APP_DESCRIPTION,
        APP_NAME,
        CAPACITANCE_UNITS,
        DEFAULT_DEVICE_AREA_CM2,
        MIN_DEVICE_AREA_CM2,
        SAMPLE_DATA_RELATIVE_PATH,
        STANDARD_CAPACITANCE_COLUMN,
        STANDARD_VOLTAGE_COLUMN,
        SUPPORTED_UPLOAD_TYPES,
    )
    from data_loader import load_csv, load_data
    
    from extraction.phase2_summary import (
        Phase2Summary,
        calculate_phase2_summary,
    )
    from plotting import plot_cv, plot_inverse_c2, plot_normalized_cv
    from preprocessing import (
        clean_data,
        convert_capacitance_units,
        validate_data,
    )
    from plot_phase1_pipeline import (
        plot_phase1_cv_regions,
        plot_phase1_linear_fit,
    )
    from utils import (
        add_normalized_capacitance,
        dataframe_to_csv_bytes,
        figure_to_png_bytes,
        format_scientific,
    )
    from extraction_new.measurement_context import (
        MeasurementContext,
        build_measurement_context,
    )

try:

    from .extraction_new.phase1_pipeline import (
        run_phase1_pipeline,
        Phase1PipelineResult,
    )

except ImportError:

    from extraction_new.phase1_pipeline import (
        run_phase1_pipeline,
        Phase1PipelineResult,
    )

StatisticsFunction = Callable[[pd.DataFrame], dict[str, int | float]]


def main() -> None:
    """Run the MOSCAP-X Streamlit application."""
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="MX",
        layout="wide",
    )

    st.title(APP_NAME)
    st.caption(APP_DESCRIPTION)

    (
        measurement_context,
        device_area_cm2,
        capacitance_unit,
        uploaded_file,
    ) = _render_sidebar()

    try:
        raw_data, source_name = _load_active_data(uploaded_file)
    except ValueError as exc:
        st.error(str(exc))
        return

    if uploaded_file is None:
        st.info(
            "Using bundled sample data. Upload a CSV or Excel file to analyze "
            "your own measurement."
        )

    st.subheader("1. Raw Data Preview")
    st.write(f"Source: `{source_name}`")
    st.dataframe(raw_data, use_container_width=True)

    st.subheader("2. Validation Results")
    validation_warnings = validate_data(raw_data)
    if validation_warnings:
        for warning in validation_warnings:
            st.warning(warning)
    else:
        st.success("Validation passed with no warnings.")

    try:
        cleaned_data = clean_data(raw_data)
        cleaned_data = convert_capacitance_units(
            cleaned_data,
            capacitance_unit,
        )
        analysis_data = add_normalized_capacitance(
            cleaned_data,
            device_area_cm2,
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    if cleaned_data.empty:
        st.warning("No valid rows remain after cleaning.")
        return

    st.subheader("3. Cleaned Data")
    st.dataframe(analysis_data, use_container_width=True)
    st.download_button(
        label="Download Cleaned CSV",
        data=dataframe_to_csv_bytes(cleaned_data),
        file_name="moscap_x_cleaned_data.csv",
        mime="text/csv",
    )

    st.subheader("4. Statistics Dashboard")
    try:
        statistics = _calculate_statistics(cleaned_data)
        _render_statistics(statistics)
    except ValueError as exc:
        st.warning(f"Statistics unavailable: {exc}")

    _render_plot_section(
        title="5. C-V Plot",
        figure_factory=lambda: plot_cv(cleaned_data),
        download_label="Download C-V PNG",
        file_name="moscap_x_cv.png",
    )
    _render_plot_section(
        title="6. Normalized C-V Plot",
        figure_factory=lambda: plot_normalized_cv(analysis_data),
        download_label="Download Normalized C-V PNG",
        file_name="moscap_x_normalized_cv.png",
    )
    _render_plot_section(
        title="7. 1/C^2-V Plot",
        figure_factory=lambda: plot_inverse_c2(cleaned_data),
        download_label="Download 1/C^2-V PNG",
        file_name="moscap_x_inverse_c2.png",
    )
    phase1b_summary = _render_phase_1b_section(
        cleaned_data,
        measurement_context,
        device_area_cm2,
    )
    _render_phase_2_section(
        cleaned_data,
        device_area_cm2,
        phase1b_summary,
    )

def _render_sidebar() -> tuple[
    MeasurementContext,
    float,
    str,
    Any,
]:
    """
    Render the application sidebar.

    Returns
    -------
    MeasurementContext
    Device area
    Capacitance unit
    Uploaded file
    """

    with st.sidebar:

        st.header("Measurement Settings")

        material = st.selectbox(
            "Material",
            (
                "Silicon",
                "Germanium",
            ),
        )

        substrate = st.selectbox(
            "Substrate Type",
            (
                "P-Type",
                "N-Type",
            ),
        )

        temperature = st.number_input(
            "Temperature (K)",
            value=300.0,
        )

        st.divider()

        st.header("Device")

        device_area_cm2 = st.number_input(
            "Device Area (cm²)",
            min_value=MIN_DEVICE_AREA_CM2,
            value=DEFAULT_DEVICE_AREA_CM2,
            format="%.6g",
        )

        capacitance_unit = st.selectbox(
            "Capacitance Unit",
            options=CAPACITANCE_UNITS,
            index=CAPACITANCE_UNITS.index("F"),
        )

        uploaded_file = st.file_uploader(
            "Upload Dataset",
            type=list(SUPPORTED_UPLOAD_TYPES),
        )

    context = build_measurement_context(
        material=material,
        substrate_type=substrate,
        
        temperature_k=temperature,
    )

    return (
        context,
        float(device_area_cm2),
        str(capacitance_unit),
        uploaded_file,
    )


def _load_active_data(uploaded_file: Any) -> tuple[pd.DataFrame, str]:
    if uploaded_file is not None:
        return load_data(uploaded_file), str(uploaded_file.name)

    sample_path = Path(__file__).resolve().parent / SAMPLE_DATA_RELATIVE_PATH
    return load_csv(sample_path), sample_path.name


def _calculate_statistics(dataframe: pd.DataFrame) -> dict[str, int | float]:
    calculate_statistics = _load_statistics_function()
    return calculate_statistics(dataframe)


def _load_statistics_function() -> StatisticsFunction:
    try:
        from .statistics import calculate_statistics

        return calculate_statistics
    except ImportError:
        module_path = Path(__file__).resolve().with_name("statistics.py")
        spec = importlib.util.spec_from_file_location(
            "moscap_x_local_statistics",
            module_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError("Unable to load local statistics module.")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.calculate_statistics


def _render_statistics(statistics: dict[str, int | float]) -> None:
    metric_items = (
        ("Total Points", str(int(statistics["total_points"]))),
        ("Voltage Min", format_scientific(statistics["voltage_min"])),
        ("Voltage Max", format_scientific(statistics["voltage_max"])),
        (
            "Capacitance Min",
            format_scientific(statistics["capacitance_min"]),
        ),
        (
            "Capacitance Max",
            format_scientific(statistics["capacitance_max"]),
        ),
        (
            "Capacitance Mean",
            format_scientific(statistics["capacitance_mean"]),
        ),
        (
            "Capacitance Std",
            format_scientific(statistics["capacitance_std"]),
        ),
    )

    first_row = st.columns(4)
    second_row = st.columns(3)
    columns = [*first_row, *second_row]

    for column, (label, value) in zip(columns, metric_items):
        column.metric(label=label, value=value)


def _render_plot_section(
    title: str,
    figure_factory: Callable[[], Figure],
    download_label: str,
    file_name: str,
) -> None:
    st.subheader(title)
    try:
        figure = figure_factory()
    except ValueError as exc:
        st.warning(f"{title} unavailable: {exc}")
        return

    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        st.pyplot(figure, clear_figure=False)
    
    st.download_button(
        label=download_label,
        data=figure_to_png_bytes(figure),
        file_name=file_name,
        mime="image/png",
    )


def _render_phase_1b_section(
    cleaned_data,
    measurement_context,
    device_area_cm2,
) -> Phase1PipelineResult | None:
   
    st.subheader("Phase 1B Parameter Extraction")
 
    try:
        voltage_array = pd.to_numeric(
            cleaned_data[STANDARD_VOLTAGE_COLUMN],
            errors="coerce",
        ).to_numpy(dtype=float)

        capacitance_array = pd.to_numeric(
            cleaned_data[STANDARD_CAPACITANCE_COLUMN],
            errors="coerce",
        ).to_numpy(dtype=float)

        summary = run_phase1_pipeline(
            voltage=voltage_array,
            capacitance=capacitance_array,
            context=measurement_context,
            area_cm2=device_area_cm2,
        )
        
    except Exception as exc:
        st.warning(f"Phase 1B parameter extraction unavailable: {exc}")
        return None

    _render_phase_1b_cards(summary)

    _render_plot_section(
        title="Automatic C-V Region Classification",
        figure_factory=lambda: plot_phase1_cv_regions(
        features=summary.features,
        plateau=summary.plateau,
        linear_region=summary.linear_region,
    ),
        download_label="Download Phase1 CV Region Plot",
        file_name="phase1_cv_regions.png",
    )

    _render_plot_section(
        title="Automatic Depletion Linear Regression",
        figure_factory=lambda: plot_phase1_linear_fit(
            dataset=summary.dataset,
            fit=summary.fit,
        ),
        download_label="Download Linear Regression Plot",
        file_name="phase1_linear_fit.png",
    )
    return summary




def _render_phase_1b_cards(
    summary: Phase1PipelineResult,
) -> None:

    row1 = st.columns(4)

    row1[0].metric(
        "Cox",
        _format_phase_1b_value(
            summary.cox.cox,
        ),
    )

    row1[1].metric(
        "V₀",
        f"{summary.vintercept.vintercept:.4f} V",
    )

    row1[2].metric(
        "R²",
        f"{summary.fit.r2:.5f}",
    )

    row1[3].metric(
        summary.doping.substrate_type,
        _format_phase_1b_value(
            summary.doping.doping_value,
        ),
    )

    row2 = st.columns(3)

    row2[0].metric(
        "Debye Length",
        _format_phase_1b_value(
            summary.debye.debye_length,
        ),
    )

    row2[1].metric(
        "CsFB",
        _format_phase_1b_value(
            summary.csfb.csfb,
        ),
    )

    row2[2].metric(
        "CFB",
        _format_phase_1b_value(
            summary.cfb.cfb,
        ),
    )

    st.metric(
        "Flat-Band Voltage",
        f"{summary.vfb.vfb:.4f} V",
    )


def _format_phase_1b_value(value: float) -> str:
    return f"{float(value):.3E}".replace("E+", "E")

def _render_phase_2_section(
    cleaned_data: pd.DataFrame,
    device_area_cm2: float,
    phase1b_summary: Phase1PipelineResult | None,
) -> None:
    st.subheader("Phase 2 MOS Parameters")
    if phase1b_summary is None:
        st.warning(
            "Phase 2 requires valid Phase 1B substrate and doping outputs."
        )
        return

    (
        metal_work_function,
        electron_affinity,
        bandgap,
        cm,
        conductance_g,
    ) = _render_phase_2_inputs()

    

    capacitance_array = pd.to_numeric(
        cleaned_data[STANDARD_CAPACITANCE_COLUMN],
        errors="coerce",
    ).to_numpy(dtype=float)

    try:
        from extraction.phase2_models import (
            Phase2Inputs,
            Phase2MaterialProperties,
        )
        

        phase2_inputs = Phase2Inputs(
            area_cm2=device_area_cm2,
            temperature_k=phase1b_summary.context.temperature_k,
            doping_cm3=phase1b_summary.doping.doping_value,
            substrate_type=phase1b_summary.context.substrate_type,
            cox_f=phase1b_summary.cox.cox,
            vfb_v=phase1b_summary.vfb.vfb,
            v0_v=phase1b_summary.vintercept.vintercept,
            phi_m_ev=metal_work_function,
        )

        phase2_materials = Phase2MaterialProperties(
            intrinsic_concentration_cm3=
                phase1b_summary.context.intrinsic_carrier_concentration_cm3,

            bandgap_ev=bandgap,

            electron_affinity_ev=electron_affinity,

            relative_permittivity=
                phase1b_summary.context.relative_permittivity,
        )

        summary = calculate_phase2_summary(
            phase2_inputs,
            phase2_materials,
        )        
    except (ArithmeticError, ValueError) as exc:
        st.warning(f"Phase 2 parameter extraction unavailable: {exc}")
        return

    _render_phase_2_metrics(summary)


def _render_phase_2_inputs():

    with st.expander(
        "Phase 2 Inputs",
        expanded=False,
    ):

        row1 = st.columns(3)
        row2 = st.columns(2)
        row3 = st.columns(2)
        row4 = st.columns(2)

        metal_work_function = row1[0].number_input(
            "Metal Work Function Φm (eV)",
            value=4.16,
        )

        electron_affinity = row1[1].number_input(
            "Electron Affinity χ (eV)",
            value=4.05,
        )
        
        bandgap = row1[2].number_input(
            "Bandgap Eg (eV)",
            value=1.12,
        )

        

       


        cm = st.number_input(
            "Cm (F)",
            value=1.78e-9,
            format="%.4e",
        )

        conductance_g = row4[1].number_input(
            "Conductance G (S)",
            value=0.0,
            format="%.4e",
        )

    return (
        float(metal_work_function),
        float(electron_affinity),
        float(bandgap),
        float(cm),
        float(conductance_g),
    )

def _render_phase_2_metrics(summary: Phase2Summary) -> None:
    metrics = (
        
        ("Diffusion Potential", summary.vd, "V"),
       
        ("Debye Length", summary.ld, "cm"),
        ("Fermi Potential", summary.phi_f, "φF"),
        ("Depletion Width", summary.wd, "cm"),

        ("Junction Electric Field", summary.em, "V/cm"),
        ("Image Force Barrier Lowering", summary.delta_phi_b, "V"),
        ("Fermi Level", summary.ef, "V"),
        ("Barrier Height", summary.phi_b, "V"),
        ("Semiconductor Capacitance", summary.cs, "F"),
        ("Flat-Band Capacitance", summary.cfb, "F"),

        ("Metal-Semiconductor Work Function", summary.phi_ms, "V"),
        ("Effective Oxide Charge", summary.qeff, "C/cm²"),
        ("Effective Charge Density", summary.neff, "cm⁻²"),
    )
    cards = [
        *st.columns(3),
        *st.columns(4),
        *st.columns(3),
        *st.columns(4),
    ]
    for card, (label, value, unit) in zip(cards, metrics):
        card.metric(label, _format_engineering(value, unit))


def _format_engineering(value: float, unit: str) -> str:
    numeric_value = float(value)
    if numeric_value == 0.0:
        return f"0 {unit}"

    magnitude = abs(numeric_value)
    if 0.1 <= magnitude < 1000.0:
        return f"{numeric_value:.4g} {unit}"

    exponent = int(np.floor(np.log10(magnitude) / 3.0) * 3)
    mantissa = numeric_value / (10.0**exponent)
    exponent_text = str(exponent) if exponent < 0 else f"+{exponent}"
    return f"{mantissa:.4g}E{exponent_text} {unit}"


if __name__ == "__main__":
    main()

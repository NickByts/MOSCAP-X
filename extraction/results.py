"""Phase 1B parameter extraction orchestration and plotting."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter

try:
    from ..constants import PLOT_DPI, PLOT_FIGURE_SIZE
    from ..fitting.linear_regression import FitResult
    from ..fitting.region_detector import (
        DEFAULT_MIN_R2,
        DEFAULT_WINDOW_PERCENTAGES,
        LinearRegionResult,
        detect_linear_region,
        fit_voltage_range,
    )
    from .doping_extractor import DopingResult, extract_doping_concentration
    from .substrate_detector import SubstrateType, detect_substrate_type
except ImportError:
    from constants import PLOT_DPI, PLOT_FIGURE_SIZE
    from fitting.linear_regression import FitResult
    from fitting.region_detector import (
        DEFAULT_MIN_R2,
        DEFAULT_WINDOW_PERCENTAGES,
        LinearRegionResult,
        detect_linear_region,
        fit_voltage_range,
    )
    from extraction.doping_extractor import (
        DopingResult,
        extract_doping_concentration,
    )
    from extraction.substrate_detector import (
        SubstrateType,
        detect_substrate_type,
    )


@dataclass(frozen=True)
class ParameterSummary:
    """Complete Phase 1B extraction result for Streamlit display."""

    region: LinearRegionResult
    fit: FitResult
    substrate_type: SubstrateType
    doping: DopingResult


def extract_parameter_summary(
    voltage_array,
    inverse_capacitance_squared,
    area_cm2,
    fit_mode: str = "Automatic",
    start_voltage: float | None = None,
    end_voltage: float | None = None,
    window_sizes: Sequence[float] = DEFAULT_WINDOW_PERCENTAGES,
    min_r2: float = DEFAULT_MIN_R2,
)-> ParameterSummary:
    
    """Detect the linear region and extract substrate and doping parameters."""

    if fit_mode == "Manual Range":

        region = fit_voltage_range(
            voltage_array=voltage_array,
            inverse_capacitance_squared=inverse_capacitance_squared,
            start_voltage=start_voltage,
            end_voltage=end_voltage,
        )

    else:

        region = detect_linear_region(
            voltage_array=voltage_array,
            inverse_capacitance_squared=inverse_capacitance_squared,
            min_r2=min_r2,
        )
    fit = FitResult(
        slope=region.slope,
        intercept=region.intercept,
        r2=region.r2,
    )
    substrate_type = detect_substrate_type(fit.slope)
    doping = extract_doping_concentration(
        slope=fit.slope,
        area_cm2=area_cm2,
        
    )

    return ParameterSummary(
        region=region,
        fit=fit,
        substrate_type=substrate_type,
        doping=doping,
    )


def plot_linear_region_detection(
    voltage_array,
    inverse_capacitance_squared,
    region,
    fit_mode: str = "Automatic",
    start_voltage: float | None = None,
    end_voltage: float | None = None,
) -> Figure:
    """Plot full 1/C^2-V data with the detected linear region overlaid."""
    voltage = _as_float_array(voltage_array, "voltage_array")
    inverse_c2 = _as_float_array(
        inverse_capacitance_squared,
        "inverse_capacitance_squared",
    )
    _validate_plot_inputs(voltage, inverse_c2, region)

    figure = Figure(figsize=PLOT_FIGURE_SIZE, dpi=PLOT_DPI)
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    full_mask = np.isfinite(voltage) & np.isfinite(inverse_c2)
    region_slice = slice(region.start_index, region.end_index + 1)
    region_voltage = voltage[region_slice]
    region_inverse_c2 = inverse_c2[region_slice]
    region_mask = np.isfinite(region_voltage) & np.isfinite(region_inverse_c2)

    axis.plot(
        voltage[full_mask],
        inverse_c2[full_mask],
        color="blue",
        linewidth=0,
        marker="o",
        markersize=1,
        label="Full 1/C²-V Curve",
    )
    axis.scatter(
        region_voltage[region_mask],
        region_inverse_c2[region_mask],
        color="red",
        s=12,
        label="Detected Linear Region",
    )
    fit_line = (
        region.slope * region_voltage
        + region.intercept
    )

    axis.plot(
        region_voltage,
        fit_line,
        color="black",
        linewidth=2,
        label="Linear Regression Fit",
    )

    if (
        fit_mode == "Manual Range"
        and start_voltage is not None
        and end_voltage is not None
    ):
        axis.axvline(
            start_voltage,
            color="green",
            linestyle="--",
            linewidth=1.5,
            label="Search Start",
        )

        axis.axvline(
            end_voltage,
            color="purple",
            linestyle="--",
            linewidth=1.5,
            label="Search End",
        )

    _style_axis(axis)
    figure.tight_layout()
    return figure


def _as_float_array(
    values: Sequence[float] | np.ndarray,
    name: str,
) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc

    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")

    return array


def _validate_plot_inputs(
    voltage: np.ndarray,
    inverse_c2: np.ndarray,
    region: LinearRegionResult,
) -> None:
    if voltage.size != inverse_c2.size:
        raise ValueError(
            "voltage_array and inverse_capacitance_squared must have "
            "equal length."
        )

    if not np.any(np.isfinite(voltage) & np.isfinite(inverse_c2)):
        raise ValueError("No finite 1/C^2-V points are available to plot.")

    if region.start_index < 0 or region.end_index >= voltage.size:
        raise ValueError("Detected region indices are outside the data bounds.")

    if region.start_index > region.end_index:
        raise ValueError("Detected region start_index is after end_index.")


def _style_axis(axis: Axes) -> None:
    axis.set_title("Linear Region Detection", fontsize=14, fontweight="bold")
    axis.set_xlabel("Voltage (V)", fontsize=12)
    axis.set_ylabel("1 / C^2 (F^-2)", fontsize=12)
    axis.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
    axis.legend(frameon=True, fontsize=10)
    axis.tick_params(axis="both", labelsize=10, direction="in")

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))
    axis.yaxis.set_major_formatter(formatter)

    for spine in axis.spines.values():
        spine.set_linewidth(1.0)

"""Visualization utilities for the new MOSCAP-X Phase 1 pipeline."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter


try:
    from .constants import (
        PLOT_DPI,
        PLOT_FIGURE_SIZE,
    )
    from .extraction_new.linear_region_detector import (
        LinearRegionResult,
    )
    from .extraction_new.curve_feature_extractor import (
        CurveFeatures,
    )

    from .extraction_new.plateau_detector import (
        DirectedPlateauResult,
    )

except ImportError:

    from constants import (
        PLOT_DPI,
        PLOT_FIGURE_SIZE,
    )

    from extraction_new.curve_feature_extractor import (
        CurveFeatures,
    )
    from extraction_new.linear_region_detector import (
        LinearRegionResult,
    )
    from extraction_new.plateau_detector import (
        DirectedPlateauResult,
    )
try:
    from .extraction_new.linear_fit_dataset_builder import (
        LinearFitDataset,
    )

    from .fitting.linear_regression import (
        FitResult,
    )

except ImportError:

    from extraction_new.linear_fit_dataset_builder import (
        LinearFitDataset,
    )

    from fitting.linear_regression import (
        FitResult,
    )


def plot_phase1_cv_regions(
    features: CurveFeatures,
    plateau: DirectedPlateauResult,
    linear_region: LinearRegionResult,
) -> Figure:
    """
    Visualize the automatic Phase-1 C-V classification.
    """

    figure, axis = _create_figure()
    voltage = features.voltage
    capacitance = features.capacitance

    # Entire measured curve
    axis.scatter(
        voltage,
        capacitance,
        s=6,
        color="lightgray",
        label="Measured C-V",
    )

    # Accumulation plateau
    plateau_region = plateau.accumulation_plateau

    axis.scatter(
        voltage[
            plateau_region.start_index:
            plateau_region.end_index + 1
        ],
        capacitance[
            plateau_region.start_index:
            plateau_region.end_index + 1
        ],
        s=10,
        color="#1f77b4",
        label="Accumulation Plateau",
    )

    # Selected linear region
    axis.scatter(
        voltage[
            linear_region.start_index:
            linear_region.end_index + 1
        ],
        capacitance[
            linear_region.start_index:
            linear_region.end_index + 1
        ],
        s=10,
        color="#d62728",
        label="Linear Region",
    )

    # Plateau boundary
    axis.axvline(
        voltage[plateau.boundary_index],
        color="#1f77b4",
        linestyle="--",
        linewidth=1.5,
        label="Plateau Boundary",
    )

    # Linear region end
    axis.axvline(
        voltage[linear_region.end_index],
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        label="Linear Region End",
    )

    _style_axis(
        axis,
        title="Automatic Plateau & Linear Region Detection",
        xlabel="Voltage (V)",
        ylabel="Capacitance (F)",
    )
    figure.tight_layout()

    return figure


def _create_figure() -> tuple[Figure, Axes]:

    figure, axis = plt.subplots(
        figsize=PLOT_FIGURE_SIZE,
        dpi=PLOT_DPI,
        constrained_layout=False,
    )

    return figure, axis


def _style_axis(
    axis: Axes,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:

    axis.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    axis.set_xlabel(
        xlabel,
        fontsize=12,
    )

    axis.set_ylabel(
        ylabel,
        fontsize=12,
    )

    axis.grid(
        True,
        which="both",
        linestyle="--",
        linewidth=0.6,
        alpha=0.7,
    )

    axis.legend()

    axis.tick_params(
        axis="both",
        labelsize=10,
        direction="in",
    )

    formatter = ScalarFormatter(
        useMathText=True,
    )

    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))

    axis.xaxis.set_major_formatter(formatter)
    axis.yaxis.set_major_formatter(formatter)

    for spine in axis.spines.values():
        spine.set_linewidth(1.0)

def plot_phase1_linear_fit(
    dataset: LinearFitDataset,
    fit: FitResult,
) -> Figure:
    """
    Visualize the depletion-region linear regression.

    Displays

    • Selected depletion points
    • Best-fit regression line
    """

    figure, axis = _create_figure()

    voltage = dataset.voltage

    inverse_c2 = dataset.inverse_c2

    axis.scatter(
        voltage,
        inverse_c2,
        s=8,
        color="red",
        label="Selected Depletion Region",
    )

    fitted_line = (
        fit.slope * voltage
        + fit.intercept
    )

    axis.plot(
        voltage,
        fitted_line,
        color="black",
        linewidth=2,
        label="Linear Regression",
    )

    _style_axis(
        axis,
        title="Automatic Depletion Region Linear Fit",
        xlabel="Voltage (V)",
        ylabel="1 / C² (F⁻²)",
    )

    axis.text(
        0.02,
        0.98,
        (
            f"Slope = {fit.slope:.3E}\n"
            f"Intercept = {fit.intercept:.3E}\n"
            f"R² = {fit.r2:.5f}"
        ),
        transform=axis.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(
            facecolor="white",
            edgecolor="gray",
            alpha=0.8,
        ),
    )

    figure.tight_layout()

    return figure
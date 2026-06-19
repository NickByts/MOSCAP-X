"""Matplotlib plotting routines for MOSCAP-X."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter

try:
    from .constants import (
        NORMALIZED_CAPACITANCE_COLUMN,
        PLOT_DPI,
        PLOT_FIGURE_SIZE,
        STANDARD_CAPACITANCE_COLUMN,
        STANDARD_VOLTAGE_COLUMN,
    )
except ImportError:
    from constants import (
        NORMALIZED_CAPACITANCE_COLUMN,
        PLOT_DPI,
        PLOT_FIGURE_SIZE,
        STANDARD_CAPACITANCE_COLUMN,
        STANDARD_VOLTAGE_COLUMN,
    )


def plot_cv(dataframe: pd.DataFrame) -> Figure:
    """Create a publication-quality capacitance-voltage plot."""
    voltage, capacitance = _prepare_xy(
        dataframe,
        STANDARD_CAPACITANCE_COLUMN,
    )
    figure, axis = _create_figure()
    axis.scatter(
        voltage,
        capacitance,
        s=1,
        color="#1f77b4",
    )
    figure.tight_layout()
    return figure


def plot_normalized_cv(dataframe: pd.DataFrame) -> Figure:
    """Create a normalized C-V plot using area normalization when present."""
    voltage = _numeric_column(dataframe, STANDARD_VOLTAGE_COLUMN)

    if NORMALIZED_CAPACITANCE_COLUMN in dataframe.columns:
        y_values = _numeric_column(dataframe, NORMALIZED_CAPACITANCE_COLUMN)
        ylabel = "Capacitance / Area (F/cm^2)"
    else:
        capacitance = _numeric_column(dataframe, STANDARD_CAPACITANCE_COLUMN)
        max_capacitance = float(np.nanmax(np.abs(capacitance)))
        if max_capacitance == 0.0:
            raise ValueError("Cannot normalize capacitance when all values are zero.")
        y_values = capacitance / max_capacitance
        ylabel = "Normalized Capacitance (C/Cmax)"

    valid_mask = np.isfinite(voltage) & np.isfinite(y_values)
    if not np.any(valid_mask):
        raise ValueError("No valid numeric data is available for plotting.")

    figure, axis = _create_figure()
    axis.plot(
        voltage[valid_mask],
        y_values[valid_mask],
        color="#2ca02c",
        linewidth=0,
        marker="o",
        markersize=1.0,
    )
    _style_axis(
        axis,
        title="Normalized C-V Characteristics",
        xlabel="Voltage (V)",
        ylabel=ylabel,
    )
    figure.tight_layout()
    return figure


def plot_inverse_c2(dataframe: pd.DataFrame) -> Figure:
    """Create a 1/C^2 versus voltage plot while excluding zero capacitance."""
    voltage = _numeric_column(dataframe, STANDARD_VOLTAGE_COLUMN)
    capacitance = _numeric_column(dataframe, STANDARD_CAPACITANCE_COLUMN)
    valid_mask = (
        np.isfinite(voltage)
        & np.isfinite(capacitance)
        & (capacitance != 0.0)
    )

    if not np.any(valid_mask):
        raise ValueError(
            "No non-zero capacitance values are available for 1/C^2 plotting."
        )

    inverse_c2 = 1.0 / np.square(capacitance[valid_mask])

    figure, axis = _create_figure()
    axis.scatter(
        voltage[valid_mask],
        inverse_c2,
        s=1,
        color="red",
    )
    _style_axis(
        axis,
        title="1/C²-V Characteristics",
        xlabel="Voltage (V)",
        ylabel="1 / C² (F⁻²)",
    )
    figure.tight_layout()
    return figure


def _prepare_xy(
    dataframe: pd.DataFrame,
    y_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    voltage = _numeric_column(dataframe, STANDARD_VOLTAGE_COLUMN)
    y_values = _numeric_column(dataframe, y_column)
    valid_mask = np.isfinite(voltage) & np.isfinite(y_values)

    if not np.any(valid_mask):
        raise ValueError("No valid numeric data is available for plotting.")

    return voltage[valid_mask], y_values[valid_mask]


def _numeric_column(dataframe: pd.DataFrame, column: str) -> np.ndarray:
    if column not in dataframe.columns:
        raise ValueError(f"Missing required column '{column}'.")

    return pd.to_numeric(dataframe[column], errors="coerce").to_numpy(
        dtype=float,
    )


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
    axis.set_title(title, fontsize=14, fontweight="bold", pad=12)
    axis.set_xlabel(xlabel, fontsize=12)
    axis.set_ylabel(ylabel, fontsize=12)
    axis.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
    axis.tick_params(axis="both", labelsize=10, direction="in")

    x_formatter = ScalarFormatter(useMathText=True)
    x_formatter.set_scientific(True)
    x_formatter.set_powerlimits((-3, 3))
    axis.xaxis.set_major_formatter(x_formatter)

    y_formatter = ScalarFormatter(useMathText=True)
    y_formatter.set_scientific(True)
    y_formatter.set_powerlimits((-3, 3))
    axis.yaxis.set_major_formatter(y_formatter)

    for spine in axis.spines.values():
        spine.set_linewidth(1.0)

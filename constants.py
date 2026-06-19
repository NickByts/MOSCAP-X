"""Application-wide constants for MOSCAP-X."""

from __future__ import annotations

APP_NAME = "MOSCAP-X"
APP_DESCRIPTION = (
    "MOS Capacitor Characterization and Parameter Extraction Platform"
)

ELEMENTARY_CHARGE_C = 1.602176634e-19

EPSILON_0_F_PER_CM = 8.854187817e-14

BOLTZMANN_CONSTANT_J_PER_K = 1.380649e-23

DEFAULT_DEVICE_AREA_CM2 = 1.0
MIN_DEVICE_AREA_CM2 = 1.0e-12

CAP_UNIT_FACTORS: dict[str, float] = {
    "F": 1.0,
    "mF": 1.0e-3,
    "uF": 1.0e-6,
    "nF": 1.0e-9,
    "pF": 1.0e-12,
}
CAPACITANCE_UNITS: tuple[str, ...] = tuple(CAP_UNIT_FACTORS.keys())

VOLTAGE_COLUMN_NAMES: tuple[str, ...] = (
    "Voltage",
    "V",
    "Bias",
    "Gate Voltage",
)
CAPACITANCE_COLUMN_NAMES: tuple[str, ...] = (
    "Capacitance",
    "C",
    "Cap",
)
CONDUCTANCE_COLUMN_NAMES: tuple[str, ...] = (
    "Conductance",
    "G",
    "Cond",
)

STANDARD_VOLTAGE_COLUMN = "Voltage"
STANDARD_CAPACITANCE_COLUMN = "Capacitance"
STANDARD_CONDUCTANCE_COLUMN = "Conductance"
NORMALIZED_CAPACITANCE_COLUMN = "Normalized Capacitance"

REQUIRED_STANDARD_COLUMNS: tuple[str, ...] = (
    STANDARD_VOLTAGE_COLUMN,
    STANDARD_CAPACITANCE_COLUMN,
)

SUPPORTED_CSV_EXTENSIONS: tuple[str, ...] = (".csv",)
SUPPORTED_EXCEL_EXTENSIONS: tuple[str, ...] = (
    ".xlsx",
    ".xlsm",
    ".xltx",
    ".xltm",
)
SUPPORTED_UPLOAD_TYPES: tuple[str, ...] = (
    "csv",
    "xlsx",
    "xlsm",
    "xltx",
    "xltm",
)

SAMPLE_DATA_RELATIVE_PATH = "sample_data/sample_data.csv"

PLOT_DPI = 150
PLOT_FIGURE_SIZE = (8.0, 6.5)

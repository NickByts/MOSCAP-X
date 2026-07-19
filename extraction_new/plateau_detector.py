"""Plateau detection for MOSCAP-X curve analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .curve_feature_extractor import CurveFeatures
    from .measurement_context import MeasurementContext
except ImportError:
    from extraction_new.curve_feature_extractor import CurveFeatures
    from extraction_new.measurement_context import MeasurementContext

DERIVATIVE_THRESHOLD_MULTIPLIER = 3.0

CONSECUTIVE_POINTS = 3

@dataclass(frozen=True, slots=True)
class PlateauCandidate:
    """Represents one candidate plateau in a C-V curve."""

    start_index: int
    end_index: int

    point_count: int

    normalized_width: float

    mean_voltage: float
    mean_capacitance: float

    standard_deviation: float

    coefficient_of_variation: float

    flatness_score: float

@dataclass(frozen=True, slots=True)
class DirectedPlateauResult:
    """
    Result returned by the physics-guided plateau detector.
    """

    accumulation_plateau: PlateauCandidate

    boundary_index: int

    confidence: float

def detect_plateaus(
    features: CurveFeatures,
    threshold_factor: float = 0.50,
    minimum_points: int = 5,
) -> list[PlateauCandidate]:
    """
    Detect all plateau candidates in the C-V curve.

    Parameters
    ----------
    features
        CurveFeatures returned by extract_curve_features().

    threshold_factor
        Multiplier applied to the median absolute derivative.

    minimum_points
        Minimum number of consecutive points required to
        qualify as a plateau.

    Returns
    -------
    list[PlateauCandidate]
        Plateau candidates in dataset order.
    """

    if threshold_factor <= 0.0:
        raise ValueError("threshold_factor must be greater than zero.")

    if minimum_points < 2:
        raise ValueError("minimum_points must be at least 2.")

    derivative = np.abs(features.first_derivative)

    median_derivative = float(np.median(derivative))

    threshold = median_derivative * threshold_factor

    plateau_mask = derivative <= threshold

    candidate_ranges = _find_connected_regions(
        plateau_mask,
        minimum_points,
    )

    candidates: list[PlateauCandidate] = []

    for start_index, end_index in candidate_ranges:

        voltage = features.voltage[
            start_index : end_index + 1
        ]

        capacitance = features.capacitance[
            start_index : end_index + 1
        ]

        mean_voltage = float(
            np.mean(voltage)
        )

        mean_capacitance = float(
            np.mean(capacitance)
        )

        standard_deviation = float(
            np.std(capacitance, ddof=1)
        )

        coefficient_of_variation = _calculate_coefficient_of_variation(
            standard_deviation,
            mean_capacitance,
        )

        flatness_score = _calculate_flatness(
            coefficient_of_variation,
        )

        normalized_width = (
            (end_index - start_index + 1)
            / features.point_count
        )

        candidates.append(
            PlateauCandidate(
            start_index=start_index,
            end_index=end_index,
            point_count=end_index - start_index + 1,
            normalized_width=normalized_width,
            mean_voltage=mean_voltage,
            mean_capacitance=mean_capacitance,
            standard_deviation=standard_deviation,
            coefficient_of_variation=coefficient_of_variation,
            flatness_score=flatness_score,
        )
        )

    return candidates


def _find_connected_regions(
    mask: np.ndarray,
    minimum_points: int,
) -> list[tuple[int, int]]:
    """Return consecutive True regions."""

    regions: list[tuple[int, int]] = []

    start = None

    for index, value in enumerate(mask):

        if value:

            if start is None:
                start = index

        else:

            if start is not None:

                end = index - 1

                if end - start + 1 >= minimum_points:
                    regions.append((start, end))

                start = None

    if start is not None:

        end = len(mask) - 1

        if end - start + 1 >= minimum_points:
            regions.append((start, end))

    return regions


def _calculate_coefficient_of_variation(
    standard_deviation: float,
    mean_capacitance: float,
) -> float:
    """Return the coefficient of variation."""

    if mean_capacitance == 0.0:
        return 0.0

    return abs(
        standard_deviation
        / mean_capacitance
    )


def _calculate_flatness(
    coefficient_of_variation: float,
) -> float:
    """Convert coefficient of variation into a flatness score."""

    return float(
        np.clip(
            1.0 - coefficient_of_variation,
            0.0,
            1.0,
        )
    )



def _scan_parameters(
    features: CurveFeatures,
    context: MeasurementContext,
) -> tuple[int, int, int]:
    """
    Return scan parameters.

    Returns
    -------
    start_index
    stop_index
    step
    """

    if context.expected_accumulation_side == "Left":
        return (
            0,
            features.point_count,
            1,
        )

    return (
        features.point_count - 1,
        -1,
        -1,
    )
def _find_transition_start(
    features: CurveFeatures,
    context: MeasurementContext,
) -> int:
    """
    Locate the boundary between the accumulation plateau
    and the transition region.

    Returns
    -------
    Boundary index.
    """

    start, stop, step = _scan_parameters(
        features,
        context,
    )

    derivative = np.abs(
        features.first_derivative
    )

    second_derivative = np.abs(
        features.second_derivative
    )

    threshold = (
        DERIVATIVE_THRESHOLD_MULTIPLIER
        * features.noise_estimate
    )

    consecutive = 0

    transition_start = None

    for index in range(
        start,
        stop,
        step,
    ):

        if derivative[index] > threshold:

            consecutive += 1

            if consecutive == CONSECUTIVE_POINTS:

                transition_start = (
                    index
                    - step * (CONSECUTIVE_POINTS - 1)
                )

                break

        else:

            consecutive = 0

    if transition_start is None:

        raise ValueError(
            "Unable to locate accumulation boundary."
        )

    return transition_start

def _refine_boundary_using_curvature(
    features: CurveFeatures,
    transition_start: int,
    step: int,
    search_window_voltage: float = 0.5,
) -> int:
    """
    Refine the accumulation boundary using the maximum
    absolute second derivative within the transition region.

    Parameters
    ----------
    features
        Curve features extracted from the C-V curve.

    transition_start
        Index where the transition begins.

    step
        Scan direction (+1 or -1).

    search_window
        Number of samples to inspect after the transition starts.

    Returns
    -------
    int
        Refined accumulation boundary index.
    """

    second_derivative = np.abs(
        features.second_derivative
    )

    point_count = features.point_count

    indices: list[int] = []

    current = transition_start

    travelled_voltage = 0.0

    while (
        0 <= current < point_count
        and travelled_voltage <= search_window_voltage
    ):
        indices.append(current)

        next_index = current + step

        if not (0 <= next_index < point_count):
            break

        travelled_voltage += abs(
            features.voltage[next_index]
            - features.voltage[current]
        )

        current = next_index

    if not indices:
        raise ValueError(
            "Unable to refine accumulation boundary."
        )

    curvature = second_derivative[indices]

    maximum_index = int(
        np.argmax(curvature)
    )

    return indices[maximum_index]

def _build_plateau(
    features: CurveFeatures,
    boundary_index: int,
    context: MeasurementContext,
) -> PlateauCandidate:
    """
    Build a PlateauCandidate from the detected
    accumulation boundary.
    """

    if context.expected_accumulation_side == "Left":

        start_index = 0
        end_index = boundary_index

    else:

        start_index = boundary_index
        end_index = features.point_count - 1

    voltage = features.voltage[
        start_index:end_index + 1
    ]

    capacitance = features.capacitance[
        start_index:end_index + 1
    ]

    point_count = end_index - start_index + 1

    if point_count < 3:
        raise ValueError(
            "Detected accumulation plateau is too small."
        )
    
    mean_voltage = float(
        np.mean(voltage)
    )

    mean_capacitance = float(
        np.mean(capacitance)
    )

    standard_deviation = float(
        np.std(
            capacitance,
            ddof=1,
        )
    )

    coefficient_of_variation = (
        _calculate_coefficient_of_variation(
            standard_deviation,
            mean_capacitance,
        )
    )

    flatness_score = (
        _calculate_flatness(
            coefficient_of_variation,
        )
    )

    normalized_width = (
        point_count
        / features.point_count
    )

    return PlateauCandidate(
        start_index=start_index,
        end_index=end_index,
        point_count=point_count,
        normalized_width=normalized_width,
        mean_voltage=mean_voltage,
        mean_capacitance=mean_capacitance,
        standard_deviation=standard_deviation,
        coefficient_of_variation=coefficient_of_variation,
        flatness_score=flatness_score,
    )

def _calculate_confidence(
    plateau: PlateauCandidate,
) -> float:
    """
    Estimate confidence in the detected
    accumulation plateau.
    """

    confidence = (
        0.70 * plateau.flatness_score
        + 0.30 * plateau.normalized_width
    )

    return float(
        np.clip(
            confidence,
            0.0,
            1.0,
        )
    )

def detect_accumulation_plateau(
    features: CurveFeatures,
    context: MeasurementContext,
) -> DirectedPlateauResult:
    """
    Detect the accumulation plateau using
    the known measurement context.
    """

    _, _, step = _scan_parameters(
        features,
        context,
    )

    transition_start = _find_transition_start(
        features,
        context,
    )

    boundary_index = _refine_boundary_using_curvature(
        features,
        transition_start,
        step,
    )

    plateau = _build_plateau(
        features,
        boundary_index,
        context,
    )

    confidence = _calculate_confidence(
        plateau,
    )
    
    return DirectedPlateauResult(
        accumulation_plateau=plateau,
        boundary_index=boundary_index,
        confidence=confidence,
    )
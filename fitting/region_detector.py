"""Automatic depletion-region detection for MOSCAP-X Phase 1B."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

try:
    from .linear_regression import FitResult, perform_linear_fit
    from .region_classifier import classify_regions
except ImportError:
    from linear_regression import FitResult, perform_linear_fit
    from region_classifier import classify_regions

DEFAULT_WINDOW_PERCENTAGES: tuple[float, ...] = (
    0.05,   # 5%
    0.08,   # 8%
    0.10,   # 10%
    0.12,   # 12%
    0.15,   # 15%
    0.20,   # 20%
)
DEFAULT_MIN_R2 = 0.98
R2_TIE_TOLERANCE = 1.0e-6

# Composite-score weights; sum is asserted below.
# Linearity measures direct agreement with the Mott-Schottky model.
_W_LINEARITY: float = 0.30
# Slope stability rewards the constant depletion slope set by doping.
_W_SLOPE_STABILITY: float = 0.20
# Curvature penalizes transition regions that bend away from depletion.
_W_CURVATURE: float = 0.15
# Monotonicity checks the physical sweep direction without flipping data.
_W_MONOTONICITY: float = 0.15
# Length favors the full depletion region over short high-R^2 fragments.
_W_LENGTH: float = 0.10
# Residual scatter catches noisy windows that R^2 alone may overrate.
_W_RESIDUAL: float = 0.05
# Span requires meaningful band-bending coverage within the measured sweep.
_W_VOLTAGE_SPAN: float = 0.05

# Fallback detector tuning.
_CV_CEILING: float = 5.0
_DECAY_K: float = 10.0
_MIN_ADAPTIVE_POINTS: int = 8
_ADAPTIVE_STEP_FRAC: float = 0.02
_EPS: float = 1e-30

# Private numeric constants used to keep new scoring helpers explicit.
_SCORE_MIN: float = 0.0
_SCORE_MAX: float = 1.0
_WINDOW_MIN_FRAC: float = 0.05
_WINDOW_MAX_FRAC: float = 0.85
_WEIGHT_SUM_TARGET: float = 1.0
_SLOPE_MIN_POINTS: int = 3
_CURVATURE_MIN_POINTS: int = 3
_CURVATURE_DIFF_ORDER: int = 2
_MONOTONICITY_MIN_POINTS: int = 2
_MIN_SPAN_FRACTION: float = 0.02
_FULL_SPAN_FRACTION: float = 0.20
_MIN_SPACING_MULTIPLIER: float = 3.0

assert np.isclose(
    _W_LINEARITY
    + _W_SLOPE_STABILITY
    + _W_CURVATURE
    + _W_MONOTONICITY
    + _W_LENGTH
    + _W_RESIDUAL
    + _W_VOLTAGE_SPAN,
    _WEIGHT_SUM_TARGET,
)


@dataclass(frozen=True)
class LinearRegionResult:
    """Detected linear depletion-region window and regression parameters."""

    start_index: int
    end_index: int
    slope: float
    intercept: float
    r2: float


@dataclass(frozen=True)
class _ScoredCandidate:
    """Internal fallback candidate with composite physics confidence."""

    start_index: int
    end_index: int
    slope: float
    intercept: float
    r2: float
    confidence: float


def _generate_window_sizes(
    point_count: int,
) -> tuple[int, ...]:
    """
    Generate legacy depletion-window sizes for the classifier-success path.

    These retain the original architecture after physical boundaries are
    available; the fallback uses adaptive physics scoring instead.
    """

    window_sizes = {
        max(10, int(point_count * percentage))
        for percentage in DEFAULT_WINDOW_PERCENTAGES
    }

    return tuple(sorted(window_sizes))


def _smooth_inverse_c2(
    inverse_c2: np.ndarray,
) -> np.ndarray:
    """
    Smooth noisy 1/C² data for locating candidate depletion regions.

    The smoothed trace is never used for final physical extraction; returned
    slope, intercept, and R² are fit from the original experimental samples.
    """

    point_count = len(inverse_c2)

    if point_count < 31:
        return inverse_c2

    return savgol_filter(
        inverse_c2,
        window_length=31,
        polyorder=3,
    )


def _calculate_local_slope(
    voltage: np.ndarray,
    inverse_c2: np.ndarray,
) -> np.ndarray:
    """
    Estimate local d(1/C²)/dV using the measured voltage spacing.

    In depletion this derivative should be approximately constant because
    the Mott-Schottky slope is set by the semiconductor doping.
    """

    return np.gradient(
        inverse_c2,
        voltage,
    )


def detect_linear_region(
    voltage_array,
    inverse_capacitance_squared,
    min_r2: float = DEFAULT_MIN_R2,
) -> LinearRegionResult:
    """Detect the physically plausible depletion segment of a 1/C^2-V curve."""
    voltage = _as_valid_array(voltage_array, "voltage_array")
    raw_inverse_c2 = _as_valid_array(
        inverse_capacitance_squared,
        "inverse_capacitance_squared",   
    )
    detection_inverse_c2 = _smooth_inverse_c2(raw_inverse_c2)

    _validate_detection_inputs(voltage, raw_inverse_c2, min_r2)

    try:
        boundaries = classify_regions(
            voltage,
            detection_inverse_c2,
        )

        print(
            "ACC_END =", boundaries.accumulation_end_index,
            "DEP_START =", boundaries.depletion_start_index,
            "DEP_END =", boundaries.depletion_end_index,
            "INV_START =", boundaries.inversion_start_index,
        )

        depletion_start = boundaries.depletion_start_index
        depletion_end = boundaries.depletion_end_index

    except ValueError:

        print(
            "Physical classifier failed. "
            "Searching the full curve."
        )

        depletion_start = 0
        depletion_end = len(voltage) - 1

    return _search_best_region(
        voltage=voltage,
        raw_inverse_c2=raw_inverse_c2,
        detection_inverse_c2=detection_inverse_c2,
        search_start=depletion_start,
        search_end=depletion_end,
        min_r2=min_r2,
    )


def fit_voltage_range(
    voltage_array,
    inverse_capacitance_squared,
    start_voltage: float,
    end_voltage: float,
) -> LinearRegionResult:
    """
    Search for the best linear region only inside a user-selected
    voltage range.
    """

    if start_voltage >= end_voltage:
        raise ValueError(
            "Start voltage must be smaller than end voltage."
        )

    voltage = _as_valid_array(
        voltage_array,
        "voltage_array",
    )

    inverse_c2 = _as_valid_array(
        inverse_capacitance_squared,
        "inverse_capacitance_squared",
    )

    mask = (
        (voltage >= start_voltage)
        &
        (voltage <= end_voltage)
    )

    selected_indices = np.where(mask)[0]

    if len(selected_indices) < 10:
        raise ValueError(
            "Selected voltage range contains too few points."
        )

    selected_voltage = voltage[mask]
    selected_inverse_c2 = inverse_c2[mask]
    selected_detection_inverse_c2 = _smooth_inverse_c2(
        selected_inverse_c2
    )

    local_region = _search_best_region(
        voltage=selected_voltage,
        raw_inverse_c2=selected_inverse_c2,
        detection_inverse_c2=selected_detection_inverse_c2,
        search_start=0,
        search_end=len(selected_voltage) - 1,
        min_r2=DEFAULT_MIN_R2,
    )

    global_start = (
        selected_indices[local_region.start_index]
    )

    global_end = (
        selected_indices[local_region.end_index]
    )

    return LinearRegionResult(
        start_index=int(global_start),
        end_index=int(global_end),
        slope=local_region.slope,
        intercept=local_region.intercept,
        r2=local_region.r2,
    )


def _search_best_region(
    voltage,
    raw_inverse_c2,
    detection_inverse_c2,
    search_start,
    search_end,
    min_r2,
) -> LinearRegionResult:
    """
    Search one bounded interval with the unified physics detector.

    The classifier supplies only the interval. This helper performs all
    candidate generation, physics scoring, and final raw-data regression.
    """

    search_lower = max(
        0,
        min(int(search_start), int(search_end)),
    )
    search_upper = min(
        len(voltage) - 1,
        max(int(search_start), int(search_end)),
    )
    search_stop = search_upper + 1
    search_voltage = voltage[search_lower:search_stop]
    search_raw_inverse_c2 = raw_inverse_c2[search_lower:search_stop]
    search_detection_inverse_c2 = detection_inverse_c2[
        search_lower:search_stop
    ]
    point_count = len(search_voltage)

    if not np.isfinite(min_r2) or min_r2 < 0.0 or min_r2 > 1.0:
        raise ValueError("min_r2 must be a finite value between 0 and 1.")

    if point_count < _MIN_ADAPTIVE_POINTS:
        raise ValueError(
            "Insufficient points for linear-region detection."
        )

    minimum_voltage_span = _minimum_candidate_span(search_voltage)
    full_credit_span = _representative_span(search_voltage)
    windows = set(
        _generate_adaptive_windows(point_count)
    )
    step = max(
        1,
        int(point_count * _ADAPTIVE_STEP_FRAC),
    )

    for window_size in _generate_window_sizes(point_count):
        if window_size > point_count:
            continue

        for start_index in range(
            0,
            point_count - window_size + 1,
            step,
        ):
            windows.add(
                (start_index, start_index + window_size)
            )

    best_result: _ScoredCandidate | None = None

    for start_index, end_index in sorted(windows):
        candidate_voltage = search_voltage[start_index:end_index]
        candidate_inverse_c2 = search_raw_inverse_c2[start_index:end_index]
        candidate_detection_inverse_c2 = search_detection_inverse_c2[
            start_index:end_index
        ]
        voltage_span = abs(
            candidate_voltage[-1] - candidate_voltage[0]
        )

        # Sliver windows do not represent meaningful depletion band bending.
        if voltage_span < minimum_voltage_span:
            continue

        fit_result = _fit_window(
            candidate_voltage,
            candidate_inverse_c2,
        )

        if fit_result is None:
            continue

        confidence = _compute_physics_score(
            candidate_voltage,
            candidate_inverse_c2,
            candidate_detection_inverse_c2,
            fit_result,
            point_count,
            full_credit_span,
        )

        candidate = _ScoredCandidate(
            start_index=search_lower + start_index,
            end_index=search_lower + end_index - 1,
            slope=fit_result.slope,
            intercept=fit_result.intercept,
            r2=fit_result.r2,
            confidence=confidence,
        )

        if best_result is None or candidate.confidence > best_result.confidence:
            best_result = candidate

    if best_result is None:
        raise ValueError(
            "Unable to locate a linear region in the dataset. "
            "All candidate windows failed the sweep-scaled voltage-span "
            "requirement."
        )

    return LinearRegionResult(
        start_index=best_result.start_index,
        end_index=best_result.end_index,
        slope=best_result.slope,
        intercept=best_result.intercept,
        r2=best_result.r2,
    )


def _fit_window(voltage: np.ndarray, inverse_c2: np.ndarray) -> FitResult | None:
    """Fit a candidate Mott-Schottky line while ignoring invalid windows."""

    try:
        return perform_linear_fit(voltage, inverse_c2)
    except ValueError:
        return None


def _is_better_candidate(
    candidate: LinearRegionResult,
    current_best: LinearRegionResult | None,
    candidate_length: int,
    current_best_length: int,
) -> bool:
    """Prefer higher-R² depletion candidates, then wider physical windows."""

    if current_best is None:
        return True

    if candidate.r2 > current_best.r2 + R2_TIE_TOLERANCE:
        return True

    similar_r2 = abs(candidate.r2 - current_best.r2) <= R2_TIE_TOLERANCE
    return similar_r2 and candidate_length > current_best_length


def _as_valid_array(
    values: Sequence[float] | np.ndarray,
    name: str,
) -> np.ndarray:
    """Return finite one-dimensional experimental data for analysis."""

    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc

    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")

    return array


def _slope_stability(
    slope_values: np.ndarray,
) -> float:
    """
    Measures how constant the local Mott-Schottky slope is.

    Smaller value indicates a more uniform depletion relation.
    """

    mean_slope = np.mean(
        np.abs(slope_values)
    )

    if mean_slope == 0:
        return np.inf

    return (
        np.std(slope_values)
        / mean_slope
    )


def _validate_detection_inputs(
    voltage: np.ndarray,
    inverse_c2: np.ndarray,
    min_r2: float,
) -> None:
    """Validate finite paired measurements before physical region search."""

    if voltage.size != inverse_c2.size:
        raise ValueError(
            "voltage_array and inverse_capacitance_squared must have "
            "equal length."
        )

    if not np.isfinite(min_r2) or min_r2 < 0.0 or min_r2 > 1.0:
        raise ValueError("min_r2 must be a finite value between 0 and 1.")


def _generate_adaptive_windows(
    point_count: int,
) -> list[tuple[int, int]]:
    """
    Generate fallback windows that expand across the measured sweep.

    The fallback samples many candidate lengths so the physics score can
    favor the largest valid depletion interval instead of a short line-like
    fragment.
    """

    min_win = max(
        _MIN_ADAPTIVE_POINTS,
        int(point_count * _WINDOW_MIN_FRAC),
    )
    max_win = min(
        point_count,
        int(point_count * _WINDOW_MAX_FRAC),
    )
    step = max(
        1,
        int(point_count * _ADAPTIVE_STEP_FRAC),
    )
    windows: list[tuple[int, int]] = []

    for i_start in range(0, point_count - min_win + 1, step):
        win = min_win

        while i_start + win <= point_count and win <= max_win:
            windows.append((i_start, i_start + win))
            win += step

    return windows


def _voltage_sweep_span(
    voltage: np.ndarray,
) -> float:
    """Return the acquired sweep span used to scale voltage criteria."""

    return max(
        float(np.ptp(voltage)),
        _EPS,
    )


def _median_voltage_spacing(
    voltage: np.ndarray,
) -> float:
    """Estimate sampling density from adjacent measured voltage points."""

    spacing = np.abs(np.diff(voltage))

    if spacing.size == 0:
        return _EPS

    finite_spacing = spacing[np.isfinite(spacing) & (spacing > _EPS)]

    if finite_spacing.size == 0:
        return _EPS

    return max(
        float(np.median(finite_spacing)),
        _EPS,
    )


def _minimum_candidate_span(
    voltage: np.ndarray,
) -> float:
    """
    Derive the minimum meaningful candidate span from the measurement.

    A depletion fit should cover more than local sampling jitter, but the
    requirement must scale with the actual sweep instead of a fixed voltage.
    """

    return max(
        _MIN_SPAN_FRACTION * _voltage_sweep_span(voltage),
        _MIN_SPACING_MULTIPLIER * _median_voltage_spacing(voltage),
    )


def _representative_span(
    voltage: np.ndarray,
) -> float:
    """
    Derive the span that receives full voltage-span score from the sweep.

    This keeps the span metric comparable for compact and wide voltage
    sweeps without assuming any absolute voltage range.
    """

    return max(
        _FULL_SPAN_FRACTION * _voltage_sweep_span(voltage),
        _minimum_candidate_span(voltage),
    )


def _clip_unit(value: float) -> float:
    """Return a finite score bounded to the unit interval."""

    if not np.isfinite(value):
        return _SCORE_MIN

    return float(np.clip(value, _SCORE_MIN, _SCORE_MAX))


def _s_linearity_score(r2: float) -> float:
    """Score adherence to the Mott-Schottky linear 1/C^2-V relation."""

    return _clip_unit(float(r2))


def _s_slope_stability_score(
    v_seg: np.ndarray,
    c_seg: np.ndarray,
) -> float:
    """
    Score constancy of d(1/C^2)/dV in the candidate depletion segment.

    The depletion slope is set by doping and should be locally stable;
    flat plateaus receive no credit because their mean slope vanishes.
    """

    if len(v_seg) < _SLOPE_MIN_POINTS:
        return _SCORE_MIN

    local_slopes = _calculate_local_slope(
        v_seg,
        c_seg,
    )
    mean_magnitude = np.mean(
        np.abs(local_slopes)
    )

    if mean_magnitude < _EPS:
        return _SCORE_MIN

    coefficient_of_variation = _slope_stability(local_slopes)

    return _clip_unit(
        _SCORE_MAX
        - coefficient_of_variation / _CV_CEILING
    )


def _s_curvature_score(
    c_seg: np.ndarray,
) -> float:
    """
    Score low second difference expected for depletion linearity.

    Transition regions bend the 1/C^2-V trace, while an ideal
    depletion window has near-zero curvature.
    """

    if len(c_seg) < _CURVATURE_MIN_POINTS:
        return _SCORE_MIN

    second_difference = np.diff(
        c_seg,
        n=_CURVATURE_DIFF_ORDER,
    )
    c_range = float(np.max(c_seg) - np.min(c_seg))
    norm = c_range if c_range > _EPS else _EPS
    mean_curvature = float(np.mean(np.abs(second_difference))) / norm

    return _clip_unit(
        _SCORE_MAX
        / (_SCORE_MAX + _DECAY_K * mean_curvature)
    )


def _s_monotonicity_score(
    c_seg: np.ndarray,
    slope: float,
) -> float:
    """
    Score whether local 1/C^2 increments follow the fitted slope sign.

    P-type depletion increases with voltage and N-type depletion decreases;
    the fitted slope supplies the direction without flipping the data.
    """

    if len(c_seg) < _MONOTONICITY_MIN_POINTS:
        return _SCORE_MIN

    delta_c = np.diff(c_seg)

    if slope >= _SCORE_MIN:
        correct = np.sum(delta_c >= _SCORE_MIN)
    else:
        correct = np.sum(delta_c <= _SCORE_MIN)

    return _clip_unit(
        float(correct)
        / len(delta_c)
    )


def _s_residual_score(
    v_seg: np.ndarray,
    c_seg: np.ndarray,
    slope: float,
    intercept: float,
) -> float:
    """
    Score absolute scatter around the fitted Mott-Schottky line.

    This complements R^2 by penalizing large residual variance even when
    relative explained variance remains high.
    """

    residuals = c_seg - (slope * v_seg + intercept)
    c_range = float(np.max(c_seg) - np.min(c_seg))
    norm = c_range if c_range > _EPS else _EPS
    residual_std = float(np.std(residuals)) / norm

    return _clip_unit(
        _SCORE_MAX
        / (_SCORE_MAX + _DECAY_K * residual_std)
    )


def _s_voltage_span_score(
    voltage_span: float,
    full_credit_span: float,
) -> float:
    """
    Score physically meaningful voltage extent for depletion extraction.

    Credit saturates at a representative fraction of the measured sweep,
    so compact and wide sweeps are judged on their own acquisition scale.
    """

    return _clip_unit(
        voltage_span
        / max(full_credit_span, _EPS)
    )


def _s_length_score(
    n_seg: int,
    n_total: int,
) -> float:
    """
    Score candidate length to prefer the full depletion section.

    Long valid windows are more likely to represent the dominant
    Mott-Schottky region than short high-R^2 fragments.
    """

    denominator = max(
        1,
        n_total - _MIN_ADAPTIVE_POINTS,
    )

    return _clip_unit(
        (n_seg - _MIN_ADAPTIVE_POINTS)
        / denominator
    )


def _compute_physics_score(
    v_seg: np.ndarray,
    c_seg: np.ndarray,
    detection_c_seg: np.ndarray,
    fit: FitResult,
    n_total: int,
    full_credit_span: float,
) -> float:
    """
    Composite physics score for a candidate window.

    Weights are grounded in the Mott-Schottky model for MOS depletion:
    R^2 and slope stability dominate because they directly measure
    obedience to the linear 1/C^2-V relation. Curvature and monotonicity
    capture physical directionality. Length, residuals, and span prevent
    selection of short artefacts.

    Returns a float in [0, 1]. Higher = more physically representative
    of the true depletion region.
    """

    voltage_span = abs(v_seg[-1] - v_seg[0])

    confidence = (
        _W_LINEARITY * _s_linearity_score(fit.r2)
        + _W_SLOPE_STABILITY
        * _s_slope_stability_score(v_seg, detection_c_seg)
        + _W_CURVATURE * _s_curvature_score(detection_c_seg)
        + _W_MONOTONICITY * _s_monotonicity_score(
            detection_c_seg,
            fit.slope,
        )
        + _W_LENGTH * _s_length_score(len(v_seg), n_total)
        + _W_RESIDUAL * _s_residual_score(
            v_seg,
            c_seg,
            fit.slope,
            fit.intercept,
        )
        + _W_VOLTAGE_SPAN * _s_voltage_span_score(
            voltage_span,
            full_credit_span,
        )
    )

    return _clip_unit(confidence)


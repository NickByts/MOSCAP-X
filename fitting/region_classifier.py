from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegionBoundaries:
    accumulation_end_index: int
    depletion_start_index: int
    depletion_end_index: int
    inversion_start_index: int


def _as_finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    """Return *values* as a validated, one-dimensional float array."""

    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real numeric values.") from exc

    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _mad(values: np.ndarray) -> float:
    """Return the median absolute deviation of a non-empty array."""

    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def _odd_window(point_count: int) -> int:
    """Choose a conservative smoothing window that scales with the data."""

    window = max(5, int(round(0.07 * point_count)))
    window = min(window, 31, point_count if point_count % 2 else point_count - 1)
    return window if window % 2 else window - 1


def _median_filter(values: np.ndarray, window: int) -> np.ndarray:
    """Suppress isolated spikes without moving sustained transitions."""

    if window <= 1:
        return values.copy()

    radius = window // 2
    padded = np.pad(values, radius, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(windows, axis=1)


def _robust_smooth(x: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Smooth data with a spike filter followed by local quadratic regression.

    The local fit uses the actual voltage spacing, so moderately non-uniform
    experimental sweeps do not acquire the derivative bias produced by a
    sample-index moving average.
    """

    point_count = values.size
    window = _odd_window(point_count)
    despiked = _median_filter(values, 3 if point_count < 50 else 5)
    radius = window // 2
    smoothed = np.empty_like(despiked)

    for index in range(point_count):
        start = max(0, index - radius)
        stop = min(point_count, index + radius + 1)

        # Keep a full window at the ends when sufficient data are available.
        if stop - start < window:
            if start == 0:
                stop = min(point_count, window)
            else:
                start = max(0, point_count - window)

        local_x = x[start:stop] - x[index]
        local_y = despiked[start:stop]
        scale = float(np.max(np.abs(local_x)))
        if scale <= np.finfo(float).eps:
            smoothed[index] = despiked[index]
            continue

        u = local_x / scale
        weights = np.maximum(0.0, 1.0 - np.abs(u) ** 3) ** 3
        design = np.column_stack((np.ones_like(u), u, u * u))

        try:
            weighted_design = design * np.sqrt(weights)[:, None]
            weighted_y = local_y * np.sqrt(weights)
            coefficients, _, _, _ = np.linalg.lstsq(
                weighted_design,
                weighted_y,
                rcond=None,
            )
            smoothed[index] = coefficients[0]
        except np.linalg.LinAlgError:
            smoothed[index] = float(np.median(local_y))

    return smoothed


def _prefix_sum(values: np.ndarray) -> np.ndarray:
    """Return a zero-prefixed cumulative sum for O(1) interval statistics."""

    return np.concatenate((np.array([0.0]), np.cumsum(values, dtype=float)))


def _interval_mean(prefix: np.ndarray, start: int, stop: int) -> float:
    return float((prefix[stop] - prefix[start]) / (stop - start))


def _linear_statistics(
    prefix_x: np.ndarray,
    prefix_y: np.ndarray,
    prefix_x2: np.ndarray,
    prefix_y2: np.ndarray,
    prefix_xy: np.ndarray,
    start: int,
    stop: int,
) -> tuple[float, float, float, float, float]:
    """Return slope, intercept, SSE, R-squared, and mean for [start, stop)."""

    count = stop - start
    sum_x = float(prefix_x[stop] - prefix_x[start])
    sum_y = float(prefix_y[stop] - prefix_y[start])
    sum_x2 = float(prefix_x2[stop] - prefix_x2[start])
    sum_y2 = float(prefix_y2[stop] - prefix_y2[start])
    sum_xy = float(prefix_xy[stop] - prefix_xy[start])

    mean_x = sum_x / count
    mean_y = sum_y / count
    centered_x2 = max(0.0, sum_x2 - count * mean_x * mean_x)
    centered_y2 = max(0.0, sum_y2 - count * mean_y * mean_y)
    centered_xy = sum_xy - count * mean_x * mean_y

    if centered_x2 <= np.finfo(float).eps:
        slope = 0.0
    else:
        slope = centered_xy / centered_x2
    intercept = mean_y - slope * mean_x
    sse = max(0.0, centered_y2 - slope * centered_xy)
    r_squared = 1.0 - sse / centered_y2 if centered_y2 > 1e-15 else 0.0
    return slope, intercept, sse, float(np.clip(r_squared, 0.0, 1.0)), mean_y


def _search_boundaries(
    coordinate: np.ndarray,
    normalized_inverse_c2: np.ndarray,
    first_derivative: np.ndarray,
    second_derivative: np.ndarray,
    starts: np.ndarray | None = None,
    ends: np.ndarray | None = None,
    relaxed: bool = False,
) -> tuple[int, int] | None:
    """Find the best physically admissible plateau-linear-plateau split."""

    count = coordinate.size
    min_plateau = max(3, int(np.ceil(0.06 * count)))
    min_depletion = max(4, int(np.ceil(0.08 * count)))
    max_depletion = max(min_depletion, int(np.floor(0.76 * count)))

    if starts is None:
        starts = np.arange(min_plateau, count - min_plateau - min_depletion + 1)
    if ends is None:
        ends = np.arange(min_plateau + min_depletion - 1, count - min_plateau)

    px = _prefix_sum(coordinate)
    py = _prefix_sum(normalized_inverse_c2)
    px2 = _prefix_sum(coordinate * coordinate)
    py2 = _prefix_sum(normalized_inverse_c2 * normalized_inverse_c2)
    pxy = _prefix_sum(coordinate * normalized_inverse_c2)
    p_abs_d1 = _prefix_sum(np.abs(first_derivative))
    p_abs_d2 = _prefix_sum(np.abs(second_derivative))

    edge_count = max(3, min(count // 5, int(np.ceil(0.12 * count))))
    edge_derivatives = np.concatenate(
        (first_derivative[:edge_count], first_derivative[-edge_count:])
    )
    derivative_noise = 1.4826 * _mad(edge_derivatives)
    derivative_reference = float(np.percentile(np.abs(first_derivative), 90.0))
    derivative_noise = min(derivative_noise, 0.25 * derivative_reference)
    derivative_noise = max(derivative_noise, 1e-10)
    p_reverse = _prefix_sum((first_derivative < -derivative_noise).astype(float))

    best_score = np.inf
    best_boundaries: tuple[int, int] | None = None
    epsilon = 1e-12

    for depletion_start in starts:
        depletion_start = int(depletion_start)
        if depletion_start < min_plateau:
            continue

        for depletion_end in ends:
            depletion_end = int(depletion_end)
            depletion_count = depletion_end - depletion_start + 1
            right_start = depletion_end + 1

            if depletion_count < min_depletion or depletion_count > max_depletion:
                continue
            if right_start > count - min_plateau:
                continue

            left = _linear_statistics(
                px, py, px2, py2, pxy, 0, depletion_start
            )
            middle = _linear_statistics(
                px, py, px2, py2, pxy, depletion_start, right_start
            )
            right = _linear_statistics(
                px, py, px2, py2, pxy, right_start, count
            )

            left_slope, _, left_sse, _, left_mean = left
            middle_slope, middle_intercept, middle_sse, middle_r2, _ = middle
            right_slope, _, right_sse, _, right_mean = right

            width = coordinate[depletion_end] - coordinate[depletion_start]
            progression = middle_slope * width
            plateau_contrast = right_mean - left_mean
            minimum_slope = max(3.0 * derivative_noise, 0.15)

            if middle_slope <= minimum_slope or progression < 0.10:
                continue
            if plateau_contrast < 0.10:
                continue

            reverse_fraction = _interval_mean(
                p_reverse, depletion_start, right_start
            )
            middle_abs_derivative = _interval_mean(
                p_abs_d1, depletion_start, right_start
            )
            left_abs_derivative = _interval_mean(p_abs_d1, 0, depletion_start)
            right_abs_derivative = _interval_mean(p_abs_d1, right_start, count)
            left_derivative_ratio = left_abs_derivative / (
                middle_abs_derivative + epsilon
            )
            right_derivative_ratio = right_abs_derivative / (
                middle_abs_derivative + epsilon
            )

            r2_limit = 0.20 if relaxed else 0.45
            reverse_limit = 0.45 if relaxed else 0.30
            derivative_ratio_limit = 1.25 if relaxed else 0.80
            if middle_r2 < r2_limit or reverse_fraction > reverse_limit:
                continue
            if (
                left_derivative_ratio > derivative_ratio_limit
                or right_derivative_ratio > derivative_ratio_limit
            ):
                continue

            left_rmse = np.sqrt(left_sse / depletion_start)
            middle_rmse = np.sqrt(middle_sse / depletion_count)
            right_rmse = np.sqrt(right_sse / (count - right_start))
            plateau_slope_ratio = (
                abs(left_slope) + abs(right_slope)
            ) / (2.0 * abs(middle_slope) + epsilon)
            curvature = _interval_mean(
                p_abs_d2, depletion_start, right_start
            )
            curvature_ratio = curvature * max(width, epsilon) / (
                abs(middle_slope) + epsilon
            )

            predicted_start = (
                middle_intercept + middle_slope * coordinate[depletion_start]
            )
            predicted_end = (
                middle_intercept + middle_slope * coordinate[depletion_end]
            )
            continuity_penalty = (
                abs(predicted_start - left_mean)
                + abs(predicted_end - right_mean)
            )
            width_fraction = depletion_count / count

            # R-squared contributes, but cannot win by itself: the score also
            # demands flat edge regions, monotonicity, low curvature, adequate
            # voltage span, and continuity with both physical plateaus.
            score = (
                1.2 * (left_rmse + right_rmse)
                + 2.8 * middle_rmse
                + 0.9 * (1.0 - middle_r2)
                + 0.8 * plateau_slope_ratio
                + 0.7 * (left_derivative_ratio + right_derivative_ratio)
                + 1.3 * reverse_fraction
                + 0.25 * curvature_ratio
                + 0.45 * continuity_penalty
                + 0.025 / max(width_fraction, 0.01)
                + 0.025 / max(progression, 0.01)
            )

            if score < best_score:
                best_score = score
                best_boundaries = (depletion_start, depletion_end)

    return best_boundaries


def _locate_depletion(
    coordinate: np.ndarray,
    normalized_inverse_c2: np.ndarray,
    first_derivative: np.ndarray,
    second_derivative: np.ndarray,
) -> tuple[int, int]:
    """Run a bounded-cost coarse search and refine it at full resolution."""

    count = coordinate.size
    max_search_points = 401

    if count <= max_search_points:
        result = _search_boundaries(
            coordinate,
            normalized_inverse_c2,
            first_derivative,
            second_derivative,
        )
        if result is None:
            result = _search_boundaries(
                coordinate,
                normalized_inverse_c2,
                first_derivative,
                second_derivative,
                relaxed=True,
            )
        if result is None:
            raise ValueError(
                "Unable to locate a physically admissible depletion region."
            )
        return result

    sample = np.unique(
        np.rint(np.linspace(0, count - 1, max_search_points)).astype(int)
    )
    coarse = _search_boundaries(
        coordinate[sample],
        normalized_inverse_c2[sample],
        first_derivative[sample],
        second_derivative[sample],
    )
    if coarse is None:
        coarse = _search_boundaries(
            coordinate[sample],
            normalized_inverse_c2[sample],
            first_derivative[sample],
            second_derivative[sample],
            relaxed=True,
        )
    if coarse is None:
        raise ValueError(
            "Unable to locate a physically admissible depletion region."
        )

    coarse_start, coarse_end = coarse
    estimated_start = int(sample[coarse_start])
    estimated_end = int(sample[coarse_end])
    search_radius = max(3, int(np.ceil(count / (max_search_points - 1))) * 2)
    starts = np.arange(
        max(3, estimated_start - search_radius),
        min(count - 1, estimated_start + search_radius) + 1,
    )
    ends = np.arange(
        max(0, estimated_end - search_radius),
        min(count - 4, estimated_end + search_radius) + 1,
    )

    refined = _search_boundaries(
        coordinate,
        normalized_inverse_c2,
        first_derivative,
        second_derivative,
        starts=starts,
        ends=ends,
    )
    if refined is None:
        refined = _search_boundaries(
            coordinate,
            normalized_inverse_c2,
            first_derivative,
            second_derivative,
            starts=starts,
            ends=ends,
            relaxed=True,
        )
    return refined if refined is not None else (estimated_start, estimated_end)


def classify_regions(
    voltage: np.ndarray,
    inverse_c2: np.ndarray,
) -> RegionBoundaries:
    """
    Classify accumulation, depletion, and inversion from a 1/C2-V curve.

    Classification is based on a constrained physical model rather than a
    fraction of the maximum derivative. The two outer regions must behave as
    low-slope plateaus, while the depletion candidate must be a significant,
    predominantly monotonic, low-curvature linear interval in 1/C2 versus V.

    P-type and N-type structures are distinguished from the low-1/C2
    accumulation plateau. For an increasing-voltage sweep it is normally on
    the negative-voltage side for P-type material and on the positive-voltage
    side for N-type material. Analysis is internally oriented from
    accumulation to inversion and then mapped back to the input indices.

    Parameters
    ----------
    voltage:
        Strictly monotonic gate-voltage samples.
    inverse_c2:
        Finite 1/C2 samples corresponding one-to-one with ``voltage``.

    Returns
    -------
    RegionBoundaries
        Boundary indices in the original input array. The depletion start and
        end are returned in ascending array-index order for compatibility with
        ordinary NumPy slicing. Accumulation and inversion indices identify
        their physically correct sides and can therefore lie on either side
        of depletion for an N-type or reverse-direction sweep.

    Raises
    ------
    ValueError
        If inputs are invalid, essentially constant, non-monotonic in voltage,
        or do not contain a defensible plateau-transition-plateau structure.
    """

    voltage_array = _as_finite_vector(voltage, "Voltage")
    inverse_c2_array = _as_finite_vector(inverse_c2, "inverse_c2")

    if voltage_array.size != inverse_c2_array.size:
        raise ValueError("Voltage and inverse_c2 must have equal length.")
    if voltage_array.size < 20:
        raise ValueError("Insufficient points for region classification.")

    voltage_steps = np.diff(voltage_array)
    if np.all(voltage_steps > 0.0):
        increasing_order = np.arange(voltage_array.size)
    elif np.all(voltage_steps < 0.0):
        increasing_order = np.arange(voltage_array.size - 1, -1, -1)
    else:
        raise ValueError(
            "Voltage must be strictly monotonic with no duplicate samples."
        )

    sorted_voltage = voltage_array[increasing_order]
    sorted_inverse_c2 = inverse_c2_array[increasing_order]
    voltage_span = float(sorted_voltage[-1] - sorted_voltage[0])
    if voltage_span <= np.finfo(float).eps:
        raise ValueError("Voltage span is too small for region classification.")

    normalized_voltage = (sorted_voltage - sorted_voltage[0]) / voltage_span
    smoothed = _robust_smooth(normalized_voltage, sorted_inverse_c2)

    low, high = np.percentile(smoothed, [5.0, 95.0])
    signal_scale = float(high - low)
    numerical_scale = max(1.0, float(np.max(np.abs(smoothed))))
    if signal_scale <= 100.0 * np.finfo(float).eps * numerical_scale:
        raise ValueError(
            "inverse_c2 has insufficient dynamic range to identify MOS regions."
        )

    edge_count = max(3, min(smoothed.size // 5, int(np.ceil(0.12 * smoothed.size))))
    left_level = float(np.median(smoothed[:edge_count]))
    right_level = float(np.median(smoothed[-edge_count:]))
    edge_difference = right_level - left_level
    edge_noise = 1.4826 * max(
        _mad(smoothed[:edge_count]),
        _mad(smoothed[-edge_count:]),
    )

    # Accumulation has the larger capacitance and therefore the lower 1/C2.
    # If edge contrast is weak, use the global robust trend only when it is
    # appreciable; otherwise polarity is not identifiable from this curve.
    orientation_threshold = max(3.0 * edge_noise, 0.03 * signal_scale)
    if abs(edge_difference) <= orientation_threshold:
        trend = _linear_statistics(
            _prefix_sum(normalized_voltage),
            _prefix_sum(smoothed),
            _prefix_sum(normalized_voltage * normalized_voltage),
            _prefix_sum(smoothed * smoothed),
            _prefix_sum(normalized_voltage * smoothed),
            0,
            smoothed.size,
        )[0]
        if abs(trend) * (normalized_voltage[-1] - normalized_voltage[0]) < (
            0.05 * signal_scale
        ):
            raise ValueError(
                "Unable to distinguish accumulation and inversion plateaus."
            )
        accumulation_on_left = trend > 0.0
    else:
        accumulation_on_left = edge_difference > 0.0

    if accumulation_on_left:
        oriented_voltage = sorted_voltage
        oriented_smoothed = smoothed
        oriented_original_indices = increasing_order
    else:
        oriented_voltage = sorted_voltage[::-1]
        oriented_smoothed = smoothed[::-1]
        oriented_original_indices = increasing_order[::-1]

    path_coordinate = np.abs(oriented_voltage - oriented_voltage[0]) / voltage_span
    center = float(np.median(oriented_smoothed))
    normalized_inverse_c2 = (oriented_smoothed - center) / signal_scale
    first_derivative = np.gradient(
        normalized_inverse_c2,
        path_coordinate,
        edge_order=2,
    )
    second_derivative = np.gradient(
        first_derivative,
        path_coordinate,
        edge_order=2,
    )

    depletion_start_oriented, depletion_end_oriented = _locate_depletion(
        path_coordinate,
        normalized_inverse_c2,
        first_derivative,
        second_derivative,
    )

    accumulation_end_oriented = depletion_start_oriented - 1
    inversion_start_oriented = depletion_end_oriented + 1
    if accumulation_end_oriented < 0 or inversion_start_oriented >= voltage_array.size:
        raise ValueError("Detected depletion region does not have two plateaus.")

    depletion_original_indices = oriented_original_indices[
        depletion_start_oriented : depletion_end_oriented + 1
    ]
    depletion_start = int(np.min(depletion_original_indices))
    depletion_end = int(np.max(depletion_original_indices))

    return RegionBoundaries(
        accumulation_end_index=int(
            oriented_original_indices[accumulation_end_oriented]
        ),
        depletion_start_index=depletion_start,
        depletion_end_index=depletion_end,
        inversion_start_index=int(oriented_original_indices[inversion_start_oriented]),
    )

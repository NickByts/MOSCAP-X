"""Tests for the MOSCAP-X Phase 1B linear regression engine."""

from __future__ import annotations

import unittest

import numpy as np

from fitting.linear_regression import perform_linear_fit


class LinearRegressionTestCase(unittest.TestCase):
    """Example regression tests for deterministic linear data."""

    def test_perform_linear_fit_returns_expected_parameters(self) -> None:
        voltage = np.arange(10.0)
        inverse_c2 = 2.5 * voltage + 4.0

        result = perform_linear_fit(voltage, inverse_c2)

        self.assertAlmostEqual(result.slope, 2.5)
        self.assertAlmostEqual(result.intercept, 4.0)
        self.assertAlmostEqual(result.r2, 1.0)

    def test_perform_linear_fit_rejects_insufficient_points(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least two points"):
            perform_linear_fit([1.0], [2.0])

    def test_perform_linear_fit_rejects_nan_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            perform_linear_fit([0.0, 1.0, 2.0], [1.0, np.nan, 3.0])


if __name__ == "__main__":
    unittest.main()

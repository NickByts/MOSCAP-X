"""Tests for automatic depletion-region detection."""

from __future__ import annotations

import unittest

import numpy as np

from fitting.region_detector import detect_linear_region


class RegionDetectorTestCase(unittest.TestCase):
    """Example tests for sliding-window linear region detection."""

    def test_detect_linear_region_prefers_longest_perfect_window(self) -> None:
        voltage = np.linspace(-5.0, 5.0, 30)
        inverse_c2 = -3.0e18 * voltage + 2.0e19

        result = detect_linear_region(voltage, inverse_c2)

        self.assertEqual(result.start_index, 0)
        self.assertEqual(result.end_index, 24)
        self.assertAlmostEqual(result.r2, 1.0)
        self.assertLess(result.slope, 0.0)

    def test_detect_linear_region_rejects_short_data(self) -> None:
        voltage = np.arange(5.0)
        inverse_c2 = voltage + 1.0

        with self.assertRaisesRegex(ValueError, "Insufficient points"):
            detect_linear_region(voltage, inverse_c2)

    def test_detect_linear_region_rejects_nan_values(self) -> None:
        voltage = np.arange(12.0)
        inverse_c2 = voltage + 1.0
        inverse_c2[3] = np.nan

        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            detect_linear_region(voltage, inverse_c2)


if __name__ == "__main__":
    unittest.main()

"""Tests for Phase 2 oxide-capacitance extraction."""

from __future__ import annotations

import unittest

import numpy as np

from extraction.cox_extractor import calculate_cox


class CoxExtractorTestCase(unittest.TestCase):
    """Validate robust accumulation-region Cox estimation."""

    def test_calculate_cox_rejects_extreme_top_outlier(self) -> None:
        capacitance = np.concatenate(
            (
                np.full(90, 1.0e-9),
                np.full(9, 2.5e-9),
                np.array([1.0e-6]),
            )
        )

        result = calculate_cox(capacitance)

        self.assertAlmostEqual(result.cox, 2.5e-9)
        self.assertEqual(result.points_used, 9)
        self.assertAlmostEqual(result.confidence, 1.0)

    def test_calculate_cox_rejects_negative_capacitance(self) -> None:
        with self.assertRaisesRegex(ValueError, "negative capacitance"):
            calculate_cox([1.0e-9, 2.0e-9, -1.0e-9, 2.1e-9, 2.2e-9])

    def test_calculate_cox_rejects_wrong_units(self) -> None:
        with self.assertRaisesRegex(ValueError, "must use 'F'"):
            calculate_cox(np.ones(10), capacitance_unit="nF")


if __name__ == "__main__":
    unittest.main()

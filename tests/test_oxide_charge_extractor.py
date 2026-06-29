"""Tests for Phase 2 effective oxide-charge extraction."""

from __future__ import annotations

import unittest

from extraction.oxide_charge_extractor import (
    calculate_effective_oxide_charge,
)


class OxideChargeExtractorTestCase(unittest.TestCase):
    """Validate oxide charge density in C/cm^2."""

    def test_calculate_effective_oxide_charge_density(self) -> None:
        result = calculate_effective_oxide_charge(
            cox_f=1.0e-9,
            vfb=-0.8,
            phi_ms=-0.7,
            area_cm2=1.0e-4,
        )

        self.assertAlmostEqual(result, -1.0e-6)

    def test_ideal_flatband_has_zero_effective_charge(self) -> None:
        result = calculate_effective_oxide_charge(
            1.0e-9,
            -0.7,
            -0.7,
            1.0e-4,
        )

        self.assertEqual(result, 0.0)

    def test_rejects_zero_area(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            calculate_effective_oxide_charge(1.0e-9, -0.8, -0.7, 0.0)


if __name__ == "__main__":
    unittest.main()

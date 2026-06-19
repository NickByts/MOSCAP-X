"""Tests for MOSCAP-X Phase 1B doping extraction."""

from __future__ import annotations

import unittest

from extraction.doping_extractor import (
    ELEMENTARY_CHARGE_C,
    EPSILON_SI_F_PER_CM,
    extract_doping_concentration,
)
from extraction.substrate_detector import detect_substrate_type


class DopingExtractorTestCase(unittest.TestCase):
    """Example tests for substrate and doping extraction."""

    def test_extracts_n_type_doping_from_negative_slope(self) -> None:
        slope = -2.7e19
        area_cm2 = 1.0e-4
        substrate_type = detect_substrate_type(slope)

        result = extract_doping_concentration(
            slope=slope,
            area_cm2=area_cm2,
            
        )
        expected = -2.0 / (
            ELEMENTARY_CHARGE_C
            * EPSILON_SI_F_PER_CM
            * area_cm2**2
            * slope
        )

        self.assertEqual(result.substrate_type, "Nd")
        self.assertEqual(result.units, "cm^-3")
        self.assertAlmostEqual(result.doping_value, expected)

    def test_extracts_p_type_doping_from_positive_slope(self) -> None:
        slope = 2.7e19
        area_cm2 = 1.0e-4
        result = extract_doping_concentration(slope, area_cm2)

        self.assertEqual(result.substrate_type, "Na")
        self.assertGreater(result.doping_value, 0.0)

    def test_rejects_zero_slope(self) -> None:
        with self.assertRaisesRegex(ValueError, "Zero slope"):
            extract_doping_concentration(0.0, 1.0e-4)


if __name__ == "__main__":
    unittest.main()

"""Tests for Phase 2 semiconductor and flat-band capacitance."""

from __future__ import annotations

import unittest

from moscap_x.extraction.capacitance_extractor import (
    calculate_flatband_capacitance,
    calculate_semiconductor_capacitance,
)
from moscap_x.extraction.phase2_validation import (
    SILICON_PERMITTIVITY_F_PER_CM,
)


class CapacitanceExtractorTestCase(unittest.TestCase):
    """Validate capacitance calculations and area safeguards."""

    def test_calculate_semiconductor_capacitance(self) -> None:
        area = 1.0e-4
        width = 3.0e-5
        expected = SILICON_PERMITTIVITY_F_PER_CM * area / width

        result = calculate_semiconductor_capacitance(area, width)

        self.assertAlmostEqual(result, expected)

    def test_calculate_flatband_capacitance(self) -> None:
        result = calculate_flatband_capacitance(2.0e-9, 1.0e-9)

        self.assertAlmostEqual(result, 2.0e-9 / 3.0)

    def test_rejects_zero_area(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            calculate_semiconductor_capacitance(0.0, 3.0e-5)


if __name__ == "__main__":
    unittest.main()

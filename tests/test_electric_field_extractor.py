"""Tests for Phase 2 maximum electric-field extraction."""

from __future__ import annotations

import unittest

from extraction.electric_field_extractor import (
    calculate_max_electric_field,
)
from extraction.phase2_validation import (
    ELEMENTARY_CHARGE_C,
    SILICON_PERMITTIVITY_F_PER_CM,
)


class ElectricFieldExtractorTestCase(unittest.TestCase):
    """Validate maximum electric field calculations in V/cm."""

    def test_calculate_max_electric_field(self) -> None:
        doping = 1.0e16
        width = 3.0e-5
        expected = (
            ELEMENTARY_CHARGE_C
            * doping
            * width
            / SILICON_PERMITTIVITY_F_PER_CM
        )

        result = calculate_max_electric_field(doping, width)

        self.assertAlmostEqual(result, expected)

    def test_rejects_zero_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            calculate_max_electric_field(1.0e16, 0.0)


if __name__ == "__main__":
    unittest.main()

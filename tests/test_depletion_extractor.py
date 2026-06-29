"""Tests for Phase 2 maximum depletion-width extraction."""

from __future__ import annotations

import unittest

import numpy as np

from extraction.depletion_extractor import (
    calculate_max_depletion_width,
)
from extraction.phase2_validation import (
    ELEMENTARY_CHARGE_C,
    SILICON_PERMITTIVITY_F_PER_CM,
)


class DepletionExtractorTestCase(unittest.TestCase):
    """Validate the Phase 2 surface-potential approximation."""

    def test_calculate_max_depletion_width(self) -> None:
        phi_f = 0.35
        doping = 1.0e16
        expected = np.sqrt(
            4.0
            * SILICON_PERMITTIVITY_F_PER_CM
            * phi_f
            / (ELEMENTARY_CHARGE_C * doping)
        )

        result = calculate_max_depletion_width(phi_f, doping)

        self.assertAlmostEqual(result, expected)

    def test_rejects_negative_doping(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            calculate_max_depletion_width(0.35, -1.0e16)


if __name__ == "__main__":
    unittest.main()

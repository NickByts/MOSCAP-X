"""Tests for Phase 2 Debye-length extraction."""

from __future__ import annotations

import unittest

import numpy as np

from moscap_x.extraction.debye_extractor import calculate_debye_length
from moscap_x.extraction.phase2_validation import (
    BOLTZMANN_CONSTANT_J_PER_K,
    ELEMENTARY_CHARGE_C,
    SILICON_PERMITTIVITY_F_PER_CM,
)


class DebyeExtractorTestCase(unittest.TestCase):
    """Validate CGS Debye length calculations."""

    def test_calculate_debye_length(self) -> None:
        expected = np.sqrt(
            SILICON_PERMITTIVITY_F_PER_CM
            * BOLTZMANN_CONSTANT_J_PER_K
            * 300.0
            / (ELEMENTARY_CHARGE_C**2 * 1.0e16)
        )

        result = calculate_debye_length(1.0e16, 300.0, "P-Type")

        self.assertAlmostEqual(result, expected)

    def test_rejects_invalid_substrate_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "N-Type.*P-Type"):
            calculate_debye_length(1.0e16, 300.0, "Intrinsic")


if __name__ == "__main__":
    unittest.main()

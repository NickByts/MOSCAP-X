"""Tests for Phase 2 flat-band voltage extraction."""

from __future__ import annotations

import unittest

from moscap_x.extraction.flatband_extractor import calculate_flatband_voltage


class FlatbandExtractorTestCase(unittest.TestCase):
    """Validate standard MOS work-function relations."""

    def test_p_type_flatband_voltage(self) -> None:
        result = calculate_flatband_voltage(
            metal_work_function_ev=4.10,
            electron_affinity_ev=4.05,
            bandgap_ev=1.12,
            phi_f=0.35,
            substrate_type="P-Type",
        )

        self.assertAlmostEqual(result.phi_ms, -0.86)
        self.assertAlmostEqual(result.vfb, -0.86)

    def test_n_type_flatband_voltage(self) -> None:
        result = calculate_flatband_voltage(4.10, 4.05, 1.12, 0.35, "N-Type")

        self.assertAlmostEqual(result.phi_ms, -0.16)
        self.assertAlmostEqual(result.vfb, -0.16)


if __name__ == "__main__":
    unittest.main()

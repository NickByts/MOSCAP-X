"""Automatic flat-band capacitance extraction for MOSCAP-X."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .cox_extractor import CoxResult
    from .csfb_extractor import CsfbResult

except ImportError:

    from extraction_new.cox_extractor import CoxResult
    from extraction_new.csfb_extractor import CsfbResult


@dataclass(frozen=True, slots=True)
class CfbResult:
    """
    Automatic flat-band capacitance extraction result.
    """

    cfb: float


def calculate_cfb(
    cox: CoxResult,
    csfb: CsfbResult,
) -> CfbResult:
    """
    Calculate the MOS flat-band capacitance.

    Formula
    -------

              Cox × CsFB
    CFB = ------------------
           Cox + CsFB
    """

    cox_value = float(
        cox.cox
    )

    csfb_value = float(
        csfb.csfb
    )

    _validate_inputs(
        cox_value,
        csfb_value,
    )

    denominator = (
        cox_value
        + csfb_value
    )

    if denominator == 0.0:
        raise ValueError(
            "Cox + CsFB equals zero."
        )

    cfb = (
        cox_value
        * csfb_value
        / denominator
    )

    if (
        not np.isfinite(cfb)
        or cfb <= 0.0
    ):
        raise ValueError(
            "Calculated CFB is non-physical."
        )

    return CfbResult(
        cfb=float(cfb),
    )


def _validate_inputs(
    cox: float,
    csfb: float,
) -> None:
    """
    Validate extractor inputs.
    """

    if not np.isfinite(cox):
        raise ValueError(
            "Cox is not finite."
        )

    if not np.isfinite(csfb):
        raise ValueError(
            "CsFB is not finite."
        )

    if cox <= 0.0:
        raise ValueError(
            "Cox must be greater than zero."
        )

    if csfb <= 0.0:
        raise ValueError(
            "CsFB must be greater than zero."
        )
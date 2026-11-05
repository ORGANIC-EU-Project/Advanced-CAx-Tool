"""
pattern_inclined.py
===================
Inclined (diagonal) line pattern generator.
"""
from __future__ import annotations

import numpy as np

from .pattern_base import DensityProfile, PatternBase


class InclinedPattern(PatternBase):
    """Single diagonal segment spanning the full column height."""

    pattern_code = "INCL"

    def _generate_points(
            self,
            x_center: float,
            y_start: float,
            y_end: float,
            pattern_amplitude: float,
            pattern_step: float,
            density_profile: DensityProfile | None,
            line_width: float,
            column_anchor_y: float | None,
            column_anchor_phase: float | None,
    ) -> np.ndarray:
        half = pattern_amplitude / 2.0
        return np.column_stack([
            np.array([x_center - half, x_center + half]),
            np.array([y_start, y_end]),
        ])



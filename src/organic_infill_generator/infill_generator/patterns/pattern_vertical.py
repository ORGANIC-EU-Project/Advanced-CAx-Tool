"""
pattern_vertical.py
===================
Simple vertical-line pattern generator.
"""
from __future__ import annotations

import numpy as np

from .pattern_base import DensityProfile, PatternBase


class VerticalPattern(PatternBase):
    """Straight vertical column used by grid-like and support patterns."""

    pattern_code = "V"

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
        return np.array([[x_center, y_start], [x_center, y_end]], dtype=float)


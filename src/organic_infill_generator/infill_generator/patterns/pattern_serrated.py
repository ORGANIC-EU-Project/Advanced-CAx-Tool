"""
pattern_serrated.py
===================
Serrated (zigzag) column patterns.

Pattern codes:
- ``"S"``: serrated
- ``"I"``: inverted serrated
"""
from __future__ import annotations

import numpy as np

from .pattern_base import DensityProfile, PatternBase, compute_step_breakpoints
from ..settings import GeometrySettings


class SerratedPattern(PatternBase):
    """Zigzag column that alternates from left to right."""

    pattern_code = "S"

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
        return _serrated_points(
            x_center=x_center,
            y_start=y_start,
            y_end=y_end,
            pattern_amplitude=pattern_amplitude,
            pattern_step=pattern_step,
            inverted=False,
        )


class InvertedSerratedPattern(PatternBase):
    """Zigzag column mirrored so the first segment starts right-to-left."""

    pattern_code = "I"

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
        return _serrated_points(
            x_center=x_center,
            y_start=y_start,
            y_end=y_end,
            pattern_amplitude=pattern_amplitude,
            pattern_step=pattern_step,
            inverted=True,
        )


def _serrated_points(
        x_center: float,
        y_start: float,
        y_end: float,
        pattern_amplitude: float,
        pattern_step: float,
        inverted: bool,
) -> np.ndarray:
    """Generate a zigzag column."""
    half = pattern_amplitude / 2.0
    left = x_center - half
    right = x_center + half
    _EPS = GeometrySettings.AREA_EPSILON

    breakpoints = compute_step_breakpoints(
        y_start=y_start, y_end=y_end, pattern_step=pattern_step,
    )

    pts: list[list[float]] = []
    going_right = not inverted

    for i in range(len(breakpoints) - 1):
        y_cur = breakpoints[i]
        y_next = breakpoints[i + 1]
        if y_next - y_cur < _EPS:
            continue

        if going_right:
            pts.append([left, y_cur])
            pts.append([right, y_next])
        else:
            pts.append([right, y_cur])
            pts.append([left, y_next])

        going_right = not going_right

    if len(pts) < 2:
        pts = [[x_center, y_start], [x_center, y_end]]

    return np.asarray(pts, dtype=float)



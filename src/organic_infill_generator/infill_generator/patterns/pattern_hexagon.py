"""
pattern_hexagon.py
==================
Honeycomb column patterns.

Pattern codes:
- ``"HH"``
- ``"HHI"``
"""
from __future__ import annotations

import numpy as np

from .pattern_base import DensityProfile, PatternBase, compute_step_breakpoints
from ..settings import GeometrySettings, HoneycombSettings


class HalfHexagonPattern(PatternBase):
    """Honeycomb half-cell column that starts on the left side."""

    pattern_code = "HH"

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
        return _honeycomb_points(
            x_center=x_center,
            y_start=y_start,
            y_end=y_end,
            pattern_amplitude=pattern_amplitude,
            pattern_step=pattern_step,
            inverted=False,
        )


class HalfHexagonInvertedPattern(PatternBase):
    """Honeycomb half-cell column mirrored so it starts on the right side."""

    pattern_code = "HHI"

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
        return _honeycomb_points(
            x_center=x_center,
            y_start=y_start,
            y_end=y_end,
            pattern_amplitude=pattern_amplitude,
            pattern_step=pattern_step,
            inverted=True,
        )


def _honeycomb_points(
        x_center: float,
        y_start: float,
        y_end: float,
        pattern_amplitude: float,
        pattern_step: float,
        inverted: bool,
) -> np.ndarray:
    """Generate one honeycomb-like column, optionally mirrored left/right."""
    half = pattern_amplitude / 2.0
    x_left = x_center - half
    x_right = x_center + half
    epsilon = GeometrySettings.AREA_EPSILON

    breakpoints = compute_step_breakpoints(
        y_start=y_start,
        y_end=y_end,
        pattern_step=pattern_step,
    )

    points: list[list[float]] = []
    for index in range(len(breakpoints) - 1):
        y_cur = breakpoints[index]
        y_next = breakpoints[index + 1]
        height = y_next - y_cur
        if height < epsilon:
            continue
        slice_height = height / HoneycombSettings.VERTICAL_DIVISIONS

        if not inverted:
            points.append([x_left, y_cur])
            points.append([x_left, y_cur + slice_height])
            points.append([x_right, y_cur + 2 * slice_height])
            points.append([x_right, y_cur + 4 * slice_height])
            points.append([x_left, y_cur + 5 * slice_height])
            points.append([x_left, y_next])
        else:
            points.append([x_right, y_cur])
            points.append([x_right, y_cur + slice_height])
            points.append([x_left, y_cur + 2 * slice_height])
            points.append([x_left, y_cur + 4 * slice_height])
            points.append([x_right, y_cur + 5 * slice_height])
            points.append([x_right, y_next])

    if len(points) < 2:
        points = [[x_center, y_start], [x_center, y_end]]

    return np.asarray(points, dtype=float)

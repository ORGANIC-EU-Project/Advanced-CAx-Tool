"""
pattern_hilbert_column.py
=========================
Hilbert space-filling curve as a column pattern.
"""
from __future__ import annotations

import numpy as np

from .pattern_base import DensityProfile, PatternBase
from ..settings import HilbertSettings


def _chaikin_smoothing(points: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Apply Chaikin smoothing while preserving the original endpoints."""
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3 or iterations <= 0:
        return pts

    for _ in range(iterations):
        new_pts = [pts[0]]
        for index in range(pts.shape[0] - 1):
            p0 = pts[index]
            p1 = pts[index + 1]
            q = HilbertSettings.CHAIKIN_Q_WEIGHT * p0 + HilbertSettings.CHAIKIN_R_WEIGHT * p1
            r = HilbertSettings.CHAIKIN_R_WEIGHT * p0 + HilbertSettings.CHAIKIN_Q_WEIGHT * p1
            new_pts.extend([q, r])
        new_pts.append(pts[-1])
        pts = np.vstack(new_pts)

    return pts


class HilbertColumnPattern(PatternBase):
    """Space-filling Hilbert-style column scaled into one pattern column."""

    pattern_code = "HILB"

    @staticmethod
    def _hilbert_d2xy(n: int, d: int) -> tuple[int, int]:
        x = y = 0
        s = 1
        while s < n:
            rx = 1 if (d & 2) else 0
            ry = 1 if ((d & 1) ^ rx) else 0
            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                x, y = y, x
            x += s * rx
            y += s * ry
            d >>= 2
            s <<= 1
        return x, y

    @staticmethod
    def _pattern_step_to_level(pattern_step: float) -> int:
        for step_threshold, level in HilbertSettings.STEP_THRESHOLDS:
            if pattern_step >= step_threshold:
                return level
        return HilbertSettings.DEFAULT_LEVEL

    @staticmethod
    def _orient_grid_points_for_column(
            grid_points: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """
        Reorient the canonical Hilbert traversal so the path spans the full
        column height from bottom to top.

        The default Hilbert ordering starts and ends on the same horizontal
        side of the square (both endpoints share the same ``y``). For column
        snake assembly we need one endpoint at the bottom and the other at the
        top, so we mirror the curve across the main diagonal.
        """
        if not grid_points:
            return grid_points

        oriented = [(gy, gx) for gx, gy in grid_points]
        if oriented[0][1] > oriented[-1][1]:
            oriented.reverse()
        return oriented

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
        """Generate a column-oriented Hilbert path for the requested span."""
        effective_step = pattern_step
        if density_profile is not None:
            mid_y = (y_start + y_end) / 2.0
            effective_step = density_profile.evaluate_pattern_step(
                x=x_center,
                y=mid_y,
                line_width=line_width,
                default_pattern_step=pattern_step,
            )

        level = self._pattern_step_to_level(effective_step)
        n = 1 << level
        total_points = n * n
        grid_points = [self._hilbert_d2xy(n, distance) for distance in range(total_points)]
        grid_points = self._orient_grid_points_for_column(grid_points)

        half = pattern_amplitude / 2.0
        x_min = x_center - half
        x_max = x_center + half
        width = x_max - x_min
        height = y_end - y_start

        if width <= 0 or height <= 0:
            return np.array([[x_center, y_start], [x_center, y_end]], dtype=float)

        max_grid = max(n - 1, 1)
        points = np.zeros((total_points, 2), dtype=float)
        for index, (gx, gy) in enumerate(grid_points):
            points[index, 0] = x_min + (gx / max_grid) * width
            points[index, 1] = y_start + (gy / max_grid) * height
        return points

    def _postprocess_points(
            self,
            points: np.ndarray,
            *,
            smoothing_passes: int = 0,
    ) -> np.ndarray:
        return _chaikin_smoothing(points, smoothing_passes)

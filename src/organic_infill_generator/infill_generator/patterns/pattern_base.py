"""
pattern_base.py
===============
Core pattern abstractions.

This module is intentionally phrased in the API vocabulary:

- ``pattern_amplitude``: horizontal width/opening of the pattern.
- ``pattern_step``: vertical distance between repeated pattern features.
  Its default value is derived from the geometry of each public pattern.
- ``column_clearance``: lateral safety margin between neighbouring columns.
- ``DensityProfile``: the density a column is drawn at.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..settings import ApiDefaults, ColumnSettings, GeometrySettings, WallSettings


# ======================================================================
# DENSITY PROFILE
# ======================================================================


@dataclass
class DensityProfile:
    """
    Uniform density over the whole part.

    The density is translated into one constant ``pattern_step`` while the
    global column layout stays unchanged.
    """

    global_density_percent: float = ApiDefaults.INFILL_DENSITY

    def __post_init__(self) -> None:
        self.global_density_percent = float(self.global_density_percent)

    def evaluate_density(self, x: float | None = None, y: float | None = None) -> float:
        """Return the density at ``(x, y)``."""
        return float(self.global_density_percent)

    def evaluate_pattern_step(
            self,
            x: float | None,
            y: float | None,
            line_width: float,
            default_pattern_step: float,
    ) -> float:
        """Return the ``pattern_step`` at ``(x, y)``."""
        return float(default_pattern_step)


# ======================================================================
# COLUMN VERTICAL EXTENT
# ======================================================================


def compute_padded_y_range(
        miny: float,
        maxy: float,
        pattern_amplitude: float,
) -> tuple[float, float]:
    """
    Compute the padded vertical extent for column generation.

    Columns extend at least 20% of the boundary height beyond each edge
    (or ``pattern_amplitude * VERTICAL_PADDING_FACTOR``, whichever is larger),
    so the pattern has room for full cycles above and below the contour.
    """
    boundary_height = maxy - miny
    y_padding = max(
        pattern_amplitude * ColumnSettings.VERTICAL_PADDING_FACTOR,
        boundary_height * ColumnSettings.MIN_VERTICAL_PADDING_RATIO,
    )
    return miny - y_padding, maxy + y_padding


# ======================================================================
# STEP BREAKPOINTS (shared by all step-based patterns)
# ======================================================================


_BREAKPOINT_DECIMALS = 2   # rounding grid of the column breakpoints


def compute_step_breakpoints(
        y_start: float,
        y_end: float,
        pattern_step: float,
) -> list[float]:
    """
    Compute the y breakpoints of a full column.

    Breakpoints fall exactly on the global grid (``y_start + n * pattern_step``)
    to prevent floating-point drift, so every column and every layer share the
    same phase.

    Returns a sorted, deduplicated list of y values from ``y_start`` up to at
    least ``y_end``. The last grid step always extends *past* ``y_end`` so no
    pattern unit is squished at the top: clipping to the actual boundary is
    handled downstream.
    """
    _EPS = GeometrySettings.AREA_EPSILON

    def _round_coord(value: float) -> float:
        return round(float(value), _BREAKPOINT_DECIMALS)

    y_start = _round_coord(y_start)
    y_end = _round_coord(y_end)
    pattern_step = max(_round_coord(pattern_step), GeometrySettings.COORD_EPSILON)

    # Continue one full step PAST y_end so the top unit is never squished.
    y_limit = y_end + pattern_step
    breakpoints: list[float] = []
    global_n = 0
    current_y = y_start

    while current_y < y_limit - _EPS:
        next_grid = y_start + global_n * pattern_step
        rounded = _round_coord(next_grid)
        if not breakpoints or abs(rounded - breakpoints[-1]) > _EPS:
            breakpoints.append(rounded)
        global_n += 1
        current_y = next_grid

    # Append the last grid point if the loop ended exactly on it.
    final_grid = y_start + global_n * pattern_step
    if not breakpoints or breakpoints[-1] < final_grid - _EPS:
        breakpoints.append(final_grid)

    return breakpoints


# ======================================================================
# PATTERN BASE CLASS
# ======================================================================


class PatternBase(ABC):
    """
    Abstract base class for all column pattern generators.

    Subclasses implement ``_generate_points`` and the base class handles
    bounding-box clipping and optional debug metadata.
    """

    pattern_code: str = ""

    def generate(
            self,
            x_center: float,
            y_start: float,
            y_end: float,
            pattern_amplitude: float = ColumnSettings.DEFAULT_PATTERN_AMPLITUDE,
            pattern_step: float = ColumnSettings.DEFAULT_PATTERN_STEP,
            density_profile: DensityProfile | None = None,
            line_width: float = ApiDefaults.LINE_WIDTH,
            max_half_width: float = ColumnSettings.DEFAULT_INFILL_SPACING * ColumnSettings.MAX_HALF_WIDTH_FACTOR,
            column_clearance: float = WallSettings.SPACING,
            column_anchor_y: float | None = None,
            column_anchor_phase: float | None = None,
            smoothing_passes: int = 0,
            debug: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict]:
        """
        Generate pattern geometry clipped to the column bounding box.
        """
        raw = self._generate_points(
            x_center=x_center,
            y_start=y_start,
            y_end=y_end,
            pattern_amplitude=pattern_amplitude,
            pattern_step=pattern_step,
            density_profile=density_profile,
            line_width=line_width,
            column_anchor_y=column_anchor_y,
            column_anchor_phase=column_anchor_phase,
        )
        raw = self._postprocess_points(raw, smoothing_passes=smoothing_passes)

        raw_half_width = 0.0
        if raw.size:
            raw_half_width = max(float(np.max(np.abs(raw[:, 0] - float(x_center)))), 0.0)

        box_half_width = max(float(max_half_width), raw_half_width)
        clip_half_width = max(box_half_width - float(column_clearance), raw_half_width)

        box_left = x_center - box_half_width
        box_right = x_center + box_half_width
        clip_left = x_center - clip_half_width
        clip_right = x_center + clip_half_width

        debug_info = None
        if debug:
            debug_info = {
                "full_box": (box_left, y_start, box_right, y_end),
                "clip_box": (clip_left, y_start, clip_right, y_end)
                if column_clearance > 0
                else None,
            }

        if raw.shape[0] >= 2 and clip_right - clip_left > GeometrySettings.COORD_EPSILON:
            raw = self._clip_to_box(raw, clip_left, clip_right)

        if debug:
            return raw, debug_info
        return raw

    def _postprocess_points(
            self,
            points: np.ndarray,
            *,
            smoothing_passes: int = 0,
    ) -> np.ndarray:
        return np.asarray(points, dtype=float)

    @abstractmethod
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
        """
        Produce the raw pattern polyline as ``(N, 2)`` array.
        """
        ...

    @staticmethod
    def _clip_to_box(pts: np.ndarray, x_left: float, x_right: float) -> np.ndarray:
        """
        Clip a polyline to the vertical range ``[x_left, x_right]``.
        """
        if pts.shape[0] < 2:
            return pts

        def _inside(x: float) -> bool:
            return x_left <= x <= x_right

        def _intersect_x(x0, y0, x1, y1, bx):
            if abs(x1 - x0) < GeometrySettings.VERTICAL_INTERSECTION_EPSILON:
                return None
            t = (bx - x0) / (x1 - x0)
            if 0.0 < t < 1.0:
                return y0 + t * (y1 - y0)
            return None

        clipped: list[list[float]] = []

        for i in range(len(pts)):
            x_cur, y_cur = pts[i]

            if i == 0:
                if _inside(x_cur):
                    clipped.append([x_cur, y_cur])
                else:
                    clipped.append([float(np.clip(x_cur, x_left, x_right)), y_cur])
                continue

            x_prev, y_prev = pts[i - 1]
            crossings: list[tuple[float, float, float]] = []
            for bx in [x_left, x_right]:
                yi = _intersect_x(x_prev, y_prev, x_cur, y_cur, bx)
                if yi is not None:
                    t = (bx - x_prev) / (x_cur - x_prev)
                    crossings.append((t, bx, yi))
            crossings.sort(key=lambda c: c[0])

            for _, bx, yi in crossings:
                clipped.append([bx, yi])

            if _inside(x_cur):
                clipped.append([x_cur, y_cur])
            else:
                clipped.append([float(np.clip(x_cur, x_left, x_right)), y_cur])

        if len(clipped) < 2:
            return np.array([[x_left, pts[0, 1]], [x_left, pts[-1, 1]]], dtype=float)

        return np.array(clipped, dtype=float)

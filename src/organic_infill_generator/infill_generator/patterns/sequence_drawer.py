"""
sequence_drawer.py
==================
Build raw infill columns from pattern generators.

Responsibilities:
- Compute column layout (x-center positions, pattern codes, spans)
- Call pattern.generate() for each column
- Return raw (unclipped) LineStrings

Clipping, fragment reconnection and path assembly are handled by clipper_module.
"""
from __future__ import annotations

from typing import Sequence

from shapely.geometry import LineString, base

from .pattern_base import DensityProfile, PatternBase, compute_padded_y_range
from .pattern_hexagon import HalfHexagonInvertedPattern, HalfHexagonPattern
from .pattern_hilbert_column import HilbertColumnPattern
from .pattern_inclined import InclinedPattern
from .pattern_serrated import InvertedSerratedPattern, SerratedPattern
from .pattern_sinusoidal import SinusoidalPattern
from .pattern_vertical import VerticalPattern
from ..settings import (
    ApiDefaults,
    ColumnSettings,
    VerticalSequenceSettings,
    WallSettings,
)


PATTERN_REGISTRY: dict[str, PatternBase] = {
    "S": SerratedPattern(),
    "I": InvertedSerratedPattern(),
    "V": VerticalPattern(),
    "MV": VerticalPattern(),
    "SIN": SinusoidalPattern(),
    "INCL": InclinedPattern(),
    "HH": HalfHexagonPattern(),
    "HHI": HalfHexagonInvertedPattern(),
    "HILB": HilbertColumnPattern(),
}


def register_pattern(code: str, pattern: PatternBase) -> None:
    """Register or replace a pattern implementation by short pattern code."""
    PATTERN_REGISTRY[code] = pattern


def _uses_modified_vertical_spacing(pattern_sequence: Sequence[str]) -> bool:
    return (
        VerticalSequenceSettings.APPLY_NARROW_VERTICAL_SPAN_IN_MIXED_SEQUENCES
        and len(set(pattern_sequence)) > 1
    )


def _column_span_for_code(
        pattern_code: str,
        pattern_sequence: Sequence[str],
        infill_spacing: float,
        line_width: float,
        column_clearance: float,
) -> float:
    use_modified_vertical = pattern_code == "MV" or (
        pattern_code == "V"
        and _uses_modified_vertical_spacing(pattern_sequence)
    )
    if use_modified_vertical:
        return max(
            line_width + VerticalSequenceSettings.MODIFIED_VERTICAL_CLEARANCE_FACTOR * column_clearance,
            line_width,
        )
    return infill_spacing


def _iter_column_layout(
        minx: float,
        maxx: float,
        pattern_sequence: Sequence[str],
        infill_spacing: float,
        line_width: float,
        column_clearance: float,
) -> list[tuple[float, str, float]]:
    """Return repeating column centers, pattern codes, and spans across X."""
    sequence_length = len(pattern_sequence)
    if sequence_length == 0:
        return []

    spans = [
        _column_span_for_code(
            pattern_code=pattern_code,
            pattern_sequence=pattern_sequence,
            infill_spacing=infill_spacing,
            line_width=line_width,
            column_clearance=column_clearance,
        )
        for pattern_code in pattern_sequence
    ]
    max_span = max(spans)

    layout: list[tuple[float, str, float]] = []
    current_index = 0
    current_code = pattern_sequence[current_index]
    current_span = spans[current_index]
    current_center = minx + ColumnSettings.FIRST_CENTER_FACTOR * current_span

    # Centers advance by half the current span plus half the next span. This
    # allows mixed-width sequences, for example narrow modified vertical columns
    # beside wider honeycomb or sinusoidal columns.
    while current_center - current_span / 2.0 <= maxx + max_span:
        layout.append((current_center, current_code, current_span))
        next_index = (current_index + 1) % sequence_length
        next_code = pattern_sequence[next_index]
        next_span = spans[next_index]
        current_center += (current_span + next_span) / 2.0
        current_index = next_index
        current_code = next_code
        current_span = next_span

    return layout


def iter_column_layout(
        minx: float,
        maxx: float,
        pattern_sequence: Sequence[str],
        infill_spacing: float,
        line_width: float,
        column_clearance: float,
) -> list[tuple[float, str, float]]:
    """Public wrapper around the column layout computation."""
    return _iter_column_layout(
        minx=minx, maxx=maxx,
        pattern_sequence=pattern_sequence,
        infill_spacing=infill_spacing,
        line_width=line_width,
        column_clearance=column_clearance,
    )


def build_infill_columns(
        infill_boundary: base.BaseGeometry,
        pattern_sequence: Sequence[str],
        infill_spacing: float = ColumnSettings.DEFAULT_INFILL_SPACING,
        pattern_amplitude: float = ColumnSettings.DEFAULT_PATTERN_AMPLITUDE,
        pattern_step: float = ColumnSettings.DEFAULT_PATTERN_STEP,
        density_profile: DensityProfile | None = None,
        line_width: float = ApiDefaults.LINE_WIDTH,
        column_clearance: float = WallSettings.SPACING,
        smoothing_passes: int = 0,
) -> list[LineString]:
    """
    Generate raw (unclipped) infill columns across ``infill_boundary``.

    The boundary is only used to determine the Y extent of the generated
    columns.  Clipping to the actual polygon boundary is the responsibility
    of clipper_module.clip_and_reconnect_columns().
    """
    if infill_boundary.is_empty:
        return []

    minx, miny, maxx, maxy = infill_boundary.bounds
    if maxx - minx <= 0:
        return []

    y_start, y_end = compute_padded_y_range(miny, maxy, pattern_amplitude)

    column_layout = _iter_column_layout(
        minx=minx,
        maxx=maxx,
        pattern_sequence=pattern_sequence,
        infill_spacing=infill_spacing,
        line_width=line_width,
        column_clearance=column_clearance,
    )

    raw_columns: list[LineString] = []
    for x_center, pattern_code, column_span in column_layout:
        pattern = PATTERN_REGISTRY.get(pattern_code)
        if pattern is None:
            valid = ", ".join(sorted(PATTERN_REGISTRY.keys()))
            raise ValueError(
                f"Unknown pattern code '{pattern_code}'. Registered patterns: {valid}"
            )

        max_half_width = column_span * ColumnSettings.MAX_HALF_WIDTH_FACTOR
        points = pattern.generate(
            x_center=x_center,
            y_start=y_start,
            y_end=y_end,
            pattern_amplitude=pattern_amplitude,
            pattern_step=pattern_step,
            density_profile=density_profile,
            line_width=line_width,
            max_half_width=max_half_width,
            column_clearance=column_clearance,
            smoothing_passes=smoothing_passes,
        )
        raw_columns.append(LineString(points))

    return raw_columns

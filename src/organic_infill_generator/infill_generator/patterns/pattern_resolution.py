"""
pattern_resolution.py
=====================
Shared slicer-like density-to-geometry resolution helpers.

This module centralizes the public density semantics so ``generator.py``, pattern
self-tests, and diagnostics all derive the same values for:

- ``line_distance``
- ``pattern_distance_factor``
- ``pattern_amplitude``
- ``pattern_step``
- ``infill_spacing``
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..settings import DensitySettings, GeometrySettings, PatternCatalog


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp ``value`` to an inclusive numeric range."""
    return max(min_value, min(max_value, value))


@dataclass(frozen=True)
class ResolvedPatternGeometry:
    """All derived spacing values needed to draw a resolved infill pattern."""

    pattern_distance_factor: float
    step_ratio: float
    base_line_distance: float
    line_distance: float
    pattern_amplitude: float
    pattern_step: float
    infill_spacing: float


def density_to_infill_spacing(line_width: float, density_percent: float) -> float:
    """Convert slicer-style density percentage into centerline spacing."""
    density_ratio = clamp(
        density_percent / 100.0,
        DensitySettings.MIN_DENSITY_RATIO,
        DensitySettings.MAX_DENSITY_RATIO,
    )
    return float(line_width / density_ratio)


def is_vertical_only_pattern(pattern_sequence: Sequence[str]) -> bool:
    """Return true when every pattern code is a vertical column primitive."""
    return bool(pattern_sequence) and set(pattern_sequence) <= PatternCatalog.VERTICAL_CODES


def pattern_distance_factor_base(infill_pattern: str, pattern_sequence: Sequence[str]) -> float:
    """Return the empirical density correction factor for a pattern family."""
    sequence_codes = set(pattern_sequence)

    if infill_pattern == "honeycomb" or sequence_codes <= PatternCatalog.HEXAGON_CODES:
        return PatternCatalog.DISTANCE_FACTORS["honeycomb"]
    if infill_pattern == "sinusoidal":
        return PatternCatalog.DISTANCE_FACTORS["sinusoidal"]
    if infill_pattern == "diamond" or sequence_codes <= PatternCatalog.DIAMOND_CODES:
        return PatternCatalog.DISTANCE_FACTORS["diamond"]
    if infill_pattern in {"serrated", "serrated_inverted"}:
        return PatternCatalog.DISTANCE_FACTORS["diamond"]
    if infill_pattern == "triangular" or (
        ("S" in sequence_codes or "I" in sequence_codes)
        and bool(sequence_codes & PatternCatalog.VERTICAL_CODES)
    ):
        return PatternCatalog.DISTANCE_FACTORS["triangular"]
    return PatternCatalog.DEFAULT_DISTANCE_FACTOR


def pattern_step_ratio(infill_pattern: str, pattern_sequence: Sequence[str]) -> float:
    """Return the step/amplitude ratio used by periodic pattern families."""
    sequence_codes = set(pattern_sequence)

    if infill_pattern == "honeycomb" or sequence_codes <= PatternCatalog.HEXAGON_CODES:
        return PatternCatalog.HEXAGON_STEP_RATIO
    if (
        infill_pattern in {"diamond", "triangular", "serrated", "serrated_inverted"}
        or (("S" in sequence_codes or "I" in sequence_codes) and sequence_codes <= PatternCatalog.TRIANGULAR_MODIFIED_CODES)
    ):
        return PatternCatalog.TRIANGULAR_STEP_RATIO
    return PatternCatalog.DEFAULT_STEP_RATIO


def pattern_step_from_amplitude(pattern_amplitude: float, step_ratio: float) -> float:
    """Compute the pattern step from amplitude and a family-specific ratio."""
    return float(pattern_amplitude * step_ratio)


def spacing_from_amplitude(
        pattern_amplitude: float,
        line_width: float,
        wall_spacing: float,
) -> float:
    """Convert visual pattern amplitude into spacing between generated columns."""
    return float(pattern_amplitude + 2.0 * wall_spacing + line_width)


def derive_defaults_from_line_distance(
        *,
        line_distance: float,
        step_ratio: float,
        pattern_sequence: Sequence[str],
        line_width: float,
        wall_spacing: float,
) -> tuple[float, float, float]:
    """Derive spacing, amplitude, and step from the target line distance."""
    if is_vertical_only_pattern(pattern_sequence):
        return line_distance, 0.0, line_distance

    if step_ratio < 1.0:
        pattern_step = line_distance
        pattern_amplitude = line_distance / max(step_ratio, GeometrySettings.AREA_EPSILON)
    else:
        pattern_amplitude = line_distance
        pattern_step = pattern_step_from_amplitude(pattern_amplitude, step_ratio)

    infill_spacing = spacing_from_amplitude(
        pattern_amplitude=pattern_amplitude,
        line_width=line_width,
        wall_spacing=wall_spacing,
    )
    return infill_spacing, pattern_amplitude, pattern_step


def line_distance_from_geometry(
        *,
        pattern_sequence: Sequence[str],
        infill_spacing: float,
        pattern_amplitude: float,
        pattern_step: float,
) -> float:
    """Estimate the effective line distance from resolved geometric settings."""
    if is_vertical_only_pattern(pattern_sequence):
        return float(infill_spacing)

    candidates = [value for value in (pattern_amplitude, pattern_step) if value > GeometrySettings.AREA_EPSILON]
    if not candidates:
        return float(infill_spacing)
    return float(min(candidates))


def estimate_density_percent(
        line_width: float,
        line_distance: float,
        pattern_distance_factor: float,
) -> float:
    """Estimate the density represented by the resolved line distance."""
    return float(
        clamp(
            line_width * pattern_distance_factor / max(line_distance, GeometrySettings.AREA_EPSILON) * 100.0,
            DensitySettings.MIN_DENSITY_PERCENT,
            DensitySettings.MAX_DENSITY_PERCENT,
        )
    )


def resolve_geometry_from_density(
        *,
        infill_pattern: str,
        pattern_sequence: Sequence[str],
        line_width: float,
        density_percent: float,
        wall_spacing: float,
) -> ResolvedPatternGeometry:
    """Resolve public density inputs into concrete pattern geometry values."""
    step_ratio = pattern_step_ratio(infill_pattern, pattern_sequence)
    distance_factor = pattern_distance_factor_base(infill_pattern, pattern_sequence)
    base_line_distance = density_to_infill_spacing(line_width, density_percent)
    line_distance = base_line_distance * distance_factor
    infill_spacing, pattern_amplitude, pattern_step = derive_defaults_from_line_distance(
        line_distance=line_distance,
        step_ratio=step_ratio,
        pattern_sequence=pattern_sequence,
        line_width=line_width,
        wall_spacing=wall_spacing,
    )
    return ResolvedPatternGeometry(
        pattern_distance_factor=float(distance_factor),
        step_ratio=float(step_ratio),
        base_line_distance=float(base_line_distance),
        line_distance=float(line_distance),
        pattern_amplitude=float(pattern_amplitude),
        pattern_step=float(pattern_step),
        infill_spacing=float(infill_spacing),
    )

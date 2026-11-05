"""Minimum-radius rule for rounding the corners of a toolpath."""
from __future__ import annotations

from dataclasses import dataclass

from ..settings import FilletSettings


@dataclass
class FilletDecision:
    """Outcome of the minimum-radius rule for a user-requested fillet."""

    requested_radius: float
    minimum_radius: float
    effective_radius: float
    applied: bool
    reason: str  # "applied" | "zero" | "below_min"


def resolve_fillet(requested_radius: float, line_width: float) -> FilletDecision:
    """
    Apply the minimum-radius rule.

    The fillet is applied only when the requested radius reaches the viable
    minimum ``MIN_RADIUS_FACTOR * line_width``. A radius of 0 or below the
    minimum disables the fillet (it is never clamped upward).
    """
    minimum = float(FilletSettings.MIN_RADIUS_FACTOR) * float(line_width)
    requested = float(requested_radius)
    if requested <= 0.0:
        return FilletDecision(requested, minimum, 0.0, False, "zero")
    if requested < minimum:
        return FilletDecision(requested, minimum, 0.0, False, "below_min")
    return FilletDecision(requested, minimum, requested, True, "applied")

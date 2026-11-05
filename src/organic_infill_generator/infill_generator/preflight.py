"""
preflight
=========

Manufacturability check run before a generation starts.

The generator is happy to produce geometry for a request the process cannot
deposit: ask for a 10 x 10 mm part with a 7 mm bead and it writes a watertight
STL holding one collapsed wall loop and no infill at all, in 30 ms, with no
indication that nothing of what was asked for is in the file. Downstream that
STL looks perfectly valid.

This module answers two questions from the resolved request and the slicer
bounds alone, before any geometry is built:

- Can the process deposit this at all? If not, generation is refused with the
  measurement that makes it impossible (:class:`InfeasibleRequestError`).
- Will it deposit what was asked for? If not, generation proceeds and the
  difference is reported (a warning-severity :class:`PreflightIssue`).

Everything here is arithmetic on a handful of floats, so the check is free
relative to a generation pass and can run unconditionally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from logging import getLogger
from typing import Any

from .settings import PreflightSettings

logger = getLogger(__name__)


#: Issue severities. Errors block generation, warnings travel with it.
ERROR = "error"
WARNING = "warning"


class InfeasibleRequestError(ValueError):
    """The requested part cannot be deposited by the configured process.

    A ``ValueError`` so it keeps the contract every other request-validation
    failure in this package already has, including the translation into
    ``AIIntegrationInfillGenerationError`` at the API boundary.
    """


@dataclass(frozen=True)
class PreflightIssue:
    """One finding, carrying the numbers that produced it."""

    code: str
    severity: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True)
class PreflightReport:
    """Every finding for one request."""

    issues: list[PreflightIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[PreflightIssue]:
        return [issue for issue in self.issues if issue.severity == ERROR]

    @property
    def warnings(self) -> list[PreflightIssue]:
        return [issue for issue in self.issues if issue.severity == WARNING]

    @property
    def feasible(self) -> bool:
        """True when nothing blocks generation. Warnings do not block."""
        return not self.errors

    def raise_if_infeasible(self) -> None:
        """Raise :class:`InfeasibleRequestError` if anything blocks generation."""
        errors = self.errors
        if not errors:
            return
        if len(errors) == 1:
            raise InfeasibleRequestError(errors[0].message)
        joined = "; ".join(issue.message for issue in errors)
        raise InfeasibleRequestError(
            f"The request cannot be manufactured ({len(errors)} problems): {joined}"
        )

    def log(self) -> None:
        """Send warnings to the logger. Errors are reported by raising."""
        for issue in self.warnings:
            logger.warning("preflight[%s] %s", issue.code, issue.message)


def _error(code: str, message: str, **detail: Any) -> PreflightIssue:
    return PreflightIssue(code=code, severity=ERROR, message=message, detail=detail)


def _warning(code: str, message: str, **detail: Any) -> PreflightIssue:
    return PreflightIssue(code=code, severity=WARNING, message=message, detail=detail)


def _check_height(resolved: Any) -> list[PreflightIssue]:
    """Requested height against the layer height that has to build it."""
    extrude_height = float(resolved.extrude_height)
    layer_height = float(resolved.layer_height)

    # 0 disables implicit 3D extrusion, so there is no height to check.
    if extrude_height <= 0 or layer_height <= 0:
        return []

    tolerance = PreflightSettings.LAYER_HEIGHT_TOLERANCE_RATIO * layer_height

    # One layer is the smallest thing the process can deposit. A height equal
    # to the layer height is exactly that, so only a height below it (beyond
    # the tolerance) is impossible.
    if extrude_height < layer_height - tolerance:
        return [
            _error(
                "height_below_one_layer",
                f"extrude_height {extrude_height:g} mm is below one layer of "
                f"{layer_height:g} mm, so not even a single layer can be "
                f"deposited. Raise extrude_height to at least "
                f"{layer_height:g} mm or lower layer_height.",
                extrude_height=extrude_height,
                layer_height=layer_height,
            )
        ]

    layers = max(1, round(extrude_height / layer_height))
    realized_height = layers * layer_height
    if abs(realized_height - extrude_height) > tolerance:
        return [
            _warning(
                "height_rounded_to_layers",
                f"extrude_height {extrude_height:g} mm is not a whole number of "
                f"{layer_height:g} mm layers: it will be built as {layers} "
                f"layer(s), {realized_height:g} mm.",
                extrude_height=extrude_height,
                layer_height=layer_height,
                layers=layers,
                realized_height=realized_height,
            )
        ]
    return []


def _check_footprint(
    resolved: Any, size_x: float, size_y: float
) -> list[PreflightIssue]:
    """Bead, walls and pattern against the XY room the part actually has."""
    issues: list[PreflightIssue] = []

    line_width = float(resolved.line_width)
    wall_spacing = float(resolved.wall_spacing)
    num_walls = int(resolved.num_walls)
    min_size = min(size_x, size_y)

    if min_size <= 0:
        return [
            _error(
                "empty_footprint",
                f"The slicer bounds have no area: {size_x:g} x {size_y:g} mm.",
                size_x=size_x,
                size_y=size_y,
            )
        ]

    if min_size < line_width:
        return [
            _error(
                "part_smaller_than_bead",
                f"The part is {min_size:g} mm across at its narrowest, thinner "
                f"than one {line_width:g} mm bead. Nothing can be deposited.",
                min_size=min_size,
                line_width=line_width,
            )
        ]

    # Same offsets the generator uses in generate_wall_centerlines: the first
    # wall centerline sits line_width / 2 in, each further one a centerline
    # step deeper, and the infill boundary is the next virtual wall after the
    # innermost one.
    centerline_step = line_width + wall_spacing

    if num_walls > 0:
        innermost_wall_inset = line_width / 2.0 + (num_walls - 1) * centerline_step
        if 2.0 * innermost_wall_inset >= min_size:
            issues.append(
                _error(
                    "walls_do_not_fit",
                    f"{num_walls} wall(s) of {line_width:g} mm need "
                    f"{2.0 * innermost_wall_inset:g} mm across a part that is "
                    f"only {min_size:g} mm wide. Reduce num_walls or "
                    f"line_width.",
                    num_walls=num_walls,
                    line_width=line_width,
                    min_size=min_size,
                    required=2.0 * innermost_wall_inset,
                )
            )
            # The infill sits inside the walls, so its own numbers say nothing
            # useful once the walls themselves have overflowed.
            return issues

    infill_inset = line_width / 2.0 + num_walls * centerline_step
    infill_extent = min_size - 2.0 * infill_inset

    if float(resolved.infill_density) <= 0:
        return issues

    if infill_extent <= 0:
        issues.append(
            _error(
                "no_infill_region",
                f"After {num_walls} wall(s) there is no room left for infill: "
                f"the {min_size:g} mm width leaves {infill_extent:g} mm. "
                f"Reduce num_walls or line_width, or enlarge the part.",
                min_size=min_size,
                num_walls=num_walls,
                line_width=line_width,
                infill_extent=infill_extent,
            )
        )
        return issues

    infill_spacing = float(resolved.infill_spacing)
    if infill_spacing > 0 and infill_extent < infill_spacing:
        issues.append(
            _error(
                "infill_below_one_line",
                f"The infill region is {infill_extent:g} mm wide but "
                f"'{resolved.infill_pattern}' at {resolved.infill_density:g}% "
                f"needs {infill_spacing:g} mm for a single pass, so no infill "
                f"would be deposited.",
                infill_extent=infill_extent,
                infill_spacing=infill_spacing,
                infill_pattern=resolved.infill_pattern,
                infill_density=float(resolved.infill_density),
            )
        )
        return issues

    if (
        infill_spacing > 0
        and resolved.infill_pattern in PreflightSettings.CELL_PATTERNS
    ):
        columns = int(infill_extent // infill_spacing)
        if columns < PreflightSettings.MIN_COLUMNS_FOR_CELL_PATTERNS:
            issues.append(
                _warning(
                    "pattern_degenerates",
                    f"'{resolved.infill_pattern}' needs at least "
                    f"{PreflightSettings.MIN_COLUMNS_FOR_CELL_PATTERNS} columns "
                    f"to close a cell but only {columns} fit in the "
                    f"{infill_extent:g} mm infill region: the pattern will come "
                    f"out as disconnected lines, not cells.",
                    infill_pattern=resolved.infill_pattern,
                    columns=columns,
                    infill_extent=infill_extent,
                    infill_spacing=infill_spacing,
                )
            )

    return issues


def _check_density(resolved: Any) -> list[PreflightIssue]:
    """Requested density against the one the resolved spacing actually reaches."""
    requested = float(resolved.infill_density)
    estimated = float(resolved.estimated_density)
    if requested <= 0:
        return []
    deviation = abs(estimated - requested)
    if deviation <= PreflightSettings.DENSITY_TOLERANCE_PERCENT:
        return []
    return [
        _warning(
            "density_not_reachable",
            f"{requested:g}% infill was requested but '{resolved.infill_pattern}' "
            f"at a {resolved.line_width:g} mm bead resolves to {estimated:g}%: "
            f"spacing is quantized by the bead.",
            requested_density=requested,
            estimated_density=estimated,
            infill_pattern=resolved.infill_pattern,
            line_width=float(resolved.line_width),
        )
    ]


def preflight_pattern_request(
    resolved: Any,
    bounds_xy: tuple[float, float, float, float],
) -> PreflightReport:
    """Check one resolved request against the part it has to fill.

    ``resolved`` is a ``ResolvedPatternRequest``; ``bounds_xy`` is the
    ``(minx, miny, maxx, maxy)`` the generator derives from the slicer output.
    Returns the report without raising, so callers can inspect a request
    without committing to generating it.
    """
    minx, miny, maxx, maxy = bounds_xy
    issues: list[PreflightIssue] = []
    issues.extend(_check_height(resolved))
    issues.extend(_check_footprint(resolved, maxx - minx, maxy - miny))
    issues.extend(_check_density(resolved))
    return PreflightReport(issues=issues)

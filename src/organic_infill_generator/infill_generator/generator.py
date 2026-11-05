"""
Public pattern-generation API entrypoint.

From a slicer output and a pattern request to the toolpath of every layer and
its exports: robot CSV, STL and, with the optional CAD backend, STEP.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import copy
import csv
import logging
import math
import os
import re

import numpy as np
import shapely
from shapely.geometry import LineString, base, box

from .models import PatternOutputOptions, PatternRequest
from .geometry.clipper_module import assemble_eulerian_path, clip_and_reconnect_columns
from .mesh.extrusion_module import (
    _build_raw_path_mesh,
    _build_raw_solid_geometry_mesh,
    _concatenate_mesh_parts,
    _finalize_mesh,
    _translate_mesh_z,
    build_solid_contour_geometry,
    build_solid_geometry_from_centerline_contour,
    export_stl,
    generate_mesh,
    generate_sandwich_mesh,
)
from ..common.geometry import apply_min_corner_radius_to_path, polygon_from_points
from .geometry.fillet import resolve_fillet
from .geometry.offset_module import generate_wall_centerlines
from .patterns.pattern_base import DensityProfile, compute_padded_y_range
from .patterns.pattern_builder import (
    ResolvedPatternRequest,
    _build_full_toolpath,
    _primary_boundary_line,
    resolve_pattern_request,
)
from .patterns.sequence_drawer import build_infill_columns, iter_column_layout
from .preflight import preflight_pattern_request
from .settings import ApiDefaults, FilletSettings, GeometrySettings, MeshSettings, PathSettings, WallSettings


logger = logging.getLogger(__name__)


@dataclass
class _PatternPreviewResult:
    """
    Internal rich result shared by plotting, mesh export, and API adapters.

    The public response intentionally exposes only a subset of this data. Keeping
    the richer object internal lets later stages reuse the same clipped path,
    wall loops, markers, and mesh hints without recomputing geometry.
    """

    contour_points: np.ndarray
    path_points: np.ndarray
    wall_centerlines: list[np.ndarray]
    final_contour: base.BaseGeometry | None
    wall_geometries: list[base.BaseGeometry] = field(default_factory=list)
    infill_boundary: base.BaseGeometry | None = None
    infill_path_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=float))
    infill_path_geometry: base.BaseGeometry | None = None
    path_geometry: base.BaseGeometry | None = None
    mesh_preview_geometry: base.BaseGeometry | None = None
    fillet_protected_indices: list[int] = field(default_factory=list)
    markers: dict[str, Any] = field(default_factory=dict)
    title: str = ""
    unit: str = ""
    num_walls: int = 0
    line_width: float = 0.0
    wall_spacing: float = 0.0
    fillet_radius: float = 0.0
    infill_density: float = 0.0
    infill_pattern: str = ""
    extrude_height: float = 0.0
    layer_height: float = 0.0
    bottom_layers: int = 0
    top_layers: int = 0
    line_distance: float = 0.0
    estimated_density: float = 0.0
    pattern_distance_factor: float = 0.0
    pattern_amplitude: float = 0.0
    pattern_step: float = 0.0
    infill_spacing: float = 0.0
    z: float | None = None
    mesh_kind: str = "infill"


@dataclass
class GlobalPatternPlan:
    """
    Pattern columns generated once in global model coordinates.
    """

    request: ResolvedPatternRequest
    bounds_xy: tuple[float, float, float, float]
    raw_columns: list[LineString]
    column_centers: list[float] = field(default_factory=list)
    column_widths: list[float] = field(default_factory=list)


def stl_to_step(stl_path: Path, step_path: Path) -> dict:
    """Convert through the CAD backend, which is imported only when asked for."""
    from ..cadquery_service import stl_to_step as convert

    return convert(stl_path, step_path)


def _results_dir(output_dir: str | None = None) -> str:
    """Create and return the directory the export functions write into."""
    results_dir = str(output_dir) if output_dir is not None else PathSettings.RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def _sanitize_name(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    return clean.strip("_") or "pattern"


def _get_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _coerce_output_options(value: dict | PatternOutputOptions | None) -> PatternOutputOptions:
    if isinstance(value, PatternOutputOptions):
        return value
    value = value or {}
    return PatternOutputOptions(
        toolpath_csv_file=bool(value.get("toolpath_csv_file", value.get("toolpath_csv", False))),
        stl_file=bool(value.get("stl_file", value.get("stl", False))),
        step_file=bool(value.get("step_file", value.get("step", False))),
        title=str(value.get("title", ApiDefaults.TITLE)),
        output_dir=value.get("output_dir"),
    )


def _bounds_xy_from_slicer_output(slicer_output: Any) -> tuple[float, float, float, float]:
    """Extract the global XY bounds from a SlicerOutput-like object."""
    output = _get_field(slicer_output, "output", slicer_output)
    model_bounds = _get_field(output, "model_bounds")
    if model_bounds is None:
        raise ValueError("slicer_output must include 'model_bounds'.")

    min_values = _get_field(model_bounds, "min")
    max_values = _get_field(model_bounds, "max")
    if min_values is None or max_values is None:
        raise ValueError("slicer_output.model_bounds must include 'min' and 'max'.")

    min_values = list(min_values)
    max_values = list(max_values)
    if len(min_values) < 2 or len(max_values) < 2:
        raise ValueError("slicer_output.model_bounds min/max must include at least x and y.")
    return (
        float(min_values[0]),
        float(min_values[1]),
        float(max_values[0]),
        float(max_values[1]),
    )


def _layers_from_slicer_output(slicer_output: Any) -> list[Any]:
    output = _get_field(slicer_output, "output", slicer_output)
    return list(_get_field(output, "layers", []) or [])


def _sorted_layers_from_slicer_output(slicer_output: Any) -> list[Any]:
    layers = _layers_from_slicer_output(slicer_output)
    return sorted(layers, key=lambda layer: float(_get_field(layer, "z", 0.0)))


def _legacy_slicer_output_from_contour(contour_points: Any, *, layer_height: float) -> dict[str, Any]:
    points = np.asarray(contour_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("'contour_points' must be an (N, 2) array-like structure.")
    if len(points) < 3:
        raise ValueError("'contour_points' must include at least three points.")

    minx, miny = np.min(points, axis=0)
    maxx, maxy = np.max(points, axis=0)
    return {
        "layer_height": float(layer_height),
        "model_bounds": {
            "min": [float(minx), float(miny), 0.0],
            "max": [float(maxx), float(maxy), float(layer_height)],
            "size": [float(maxx - minx), float(maxy - miny), float(layer_height)],
        },
        "layers": [
            {
                "z": 0.0,
                "method": "legacy",
                "contours": [
                    {
                        "id": 0,
                        "area": float(abs(polygon_from_points(points).area)),
                        "is_largest": True,
                        "exterior": points.tolist(),
                        "holes": [],
                    }
                ],
                "open_lines": [],
            }
        ],
        "total_layers": 1,
        "closed_layers": 1,
        "failed_layers": 0,
        "noise_layers": 0,
    }


def _selected_layer_contour(layer: Any) -> Any | None:
    contours = list(_get_field(layer, "contours", []) or [])
    if not contours:
        return None

    for contour in contours:
        if bool(_get_field(contour, "is_largest", False)):
            return contour
    return max(contours, key=lambda item: float(_get_field(item, "area", 0.0)))


def _contour_points_from_layer(layer: Any) -> np.ndarray | None:
    contour = _selected_layer_contour(layer)
    if contour is None:
        return None
    exterior = _get_field(contour, "exterior")
    if exterior is None:
        return None
    points = np.asarray(exterior, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        return None
    return points


def build_global_pattern_plan(
        request: dict | PatternRequest | ResolvedPatternRequest,
        slicer_output: Any,
) -> GlobalPatternPlan:
    """
    Generate raw pattern columns once over the slicer's global XY bounds.
    """
    resolved = request if isinstance(request, ResolvedPatternRequest) else resolve_pattern_request(request)

    # Apply the minimum-viable-fillet rule once, centrally, so every consumer of
    # this plan sees the same self-consistent radius: a request below the
    # minimum turn radius (or exactly 0) disables the fillet rather than being
    # forced up.
    resolved.fillet_radius = resolve_fillet(
        resolved.fillet_radius, resolved.line_width,
    ).effective_radius

    minx, miny, maxx, maxy = _bounds_xy_from_slicer_output(slicer_output)

    # Refuse what the process cannot deposit before building any geometry, and
    # report what it can deposit but not as asked. Pure arithmetic on the
    # resolved request, so it costs nothing next to the pass it guards.
    report = preflight_pattern_request(resolved, (minx, miny, maxx, maxy))
    report.raise_if_infeasible()
    report.log()

    global_boundary = box(minx, miny, maxx, maxy)

    # Column positions are global, not per-layer. This keeps multilayer output
    # visually coherent: each layer clips the same base columns to its contour.
    y_range = compute_padded_y_range(miny, maxy, resolved.pattern_amplitude)
    column_layout = iter_column_layout(
        minx=minx,
        maxx=maxx,
        pattern_sequence=resolved.pattern_sequence,
        infill_spacing=resolved.infill_spacing,
        line_width=resolved.line_width,
        column_clearance=resolved.wall_spacing,
    )
    col_centers = [x for x, _, _ in column_layout]
    col_widths = [span for _, _, span in column_layout]

    raw_columns = build_infill_columns(
        infill_boundary=global_boundary,
        pattern_sequence=resolved.pattern_sequence,
        infill_spacing=resolved.infill_spacing,
        pattern_amplitude=resolved.pattern_amplitude,
        pattern_step=resolved.pattern_step,
        density_profile=DensityProfile(global_density_percent=resolved.estimated_density),
        line_width=resolved.line_width,
        column_clearance=resolved.wall_spacing,
        smoothing_passes=0,
    )
    return GlobalPatternPlan(
        request=resolved,
        bounds_xy=(minx, miny, maxx, maxy),
        raw_columns=raw_columns,
        column_centers=col_centers,
        column_widths=col_widths,
    )


def _build_boundary_fitted_vertical_columns(
        infill_boundary: base.BaseGeometry,
        line_width: float,
        max_overlap_ratio: float = PathSettings.SOLID_RASTER_MAX_OVERLAP_RATIO,
) -> list[LineString]:
    """Build a fully dense vertical raster fitted to both X boundary limits.

    This helper is intentionally reserved for the temporary solid-layer CSV
    previews.  The first and last centerlines sit just inside the infill
    boundary by a numeric tolerance.  Interior gaps retain the exact requested
    ``line_width`` pitch; only the minimum symmetric number of edge-adjacent
    gaps is compressed, and never by more than ``max_overlap_ratio``.

    If the remainder cannot be absorbed without exceeding that limit, the
    constant-width raster is geometrically infeasible.  Failing explicitly is
    safer than silently over-extruding because the current CSV schema has no
    per-segment extrusion-width or flow command for Arachne-style gap filling.
    """
    if infill_boundary.is_empty or line_width <= GeometrySettings.LENGTH_EPSILON:
        return []
    if not 0.0 < max_overlap_ratio < 1.0:
        raise ValueError("'max_overlap_ratio' must be between 0 and 1.")

    minx, miny, maxx, maxy = map(float, infill_boundary.bounds)
    width = maxx - minx
    if width <= GeometrySettings.LENGTH_EPSILON:
        return []

    tolerance = min(
        width / 4.0,
        max(2.0 * GeometrySettings.LENGTH_EPSILON, line_width * 1e-9),
    )
    if width <= line_width + GeometrySettings.LENGTH_EPSILON:
        centers = np.asarray([(minx + maxx) / 2.0], dtype=float)
    else:
        interval_count = max(
            1,
            int(math.ceil(
                (width - GeometrySettings.LENGTH_EPSILON) / line_width
            )),
        )
        total_overlap = max(interval_count * line_width - width, 0.0)
        gaps = np.full(interval_count, line_width, dtype=float)

        if total_overlap > GeometrySettings.LENGTH_EPSILON:
            max_overlap = line_width * max_overlap_ratio
            correction_count = int(math.ceil(
                (total_overlap - GeometrySettings.LENGTH_EPSILON) / max_overlap
            ))
            # Use equal-size correction bands on both sides.  If every gap is
            # needed and their count is odd, uniform correction is the only
            # symmetric constant-width solution.
            if correction_count % 2 and correction_count < interval_count:
                correction_count += 1
            if correction_count > interval_count:
                minimum_overlap = total_overlap / interval_count
                raise ValueError(
                    "Solid raster cannot touch both infill boundaries with "
                    f"line_width={line_width:.6g}: the minimum required "
                    f"overlap is {minimum_overlap:.6g} "
                    f"({minimum_overlap / line_width:.1%}), above the "
                    f"configured {max_overlap_ratio:.1%} limit. The current "
                    "CSV format needs variable width/flow to fill this layer."
                )

            overlap_per_gap = total_overlap / correction_count
            left_count = correction_count // 2
            right_count = correction_count - left_count
            gaps[:left_count] -= overlap_per_gap
            gaps[interval_count - right_count:] -= overlap_per_gap

        centers = minx + np.concatenate((
            np.asarray([0.0], dtype=float),
            np.cumsum(gaps),
        ))
        # Keep boundary-coincident lines numerically inside the polygon so
        # Shapely cannot collapse them during intersection.
        centers[0] += tolerance
        centers[-1] -= tolerance

    y_start, y_end = compute_padded_y_range(miny, maxy, pattern_amplitude=0.0)
    return [
        LineString(((float(x), y_start), (float(x), y_end)))
        for x in centers
    ]


def _largest_infill_island(infill_boundary: base.BaseGeometry) -> base.BaseGeometry:
    """Keep the largest island of an infill boundary the offset split apart.

    Offsetting a contour with a narrow neck leaves several islands. One
    continuous path cannot visit them without crossing empty space, so this
    first pass fills the largest, as it does with the contours of a layer.
    """
    if infill_boundary.geom_type != "MultiPolygon":
        return infill_boundary
    islands = sorted(infill_boundary.geoms, key=lambda island: island.area, reverse=True)
    logger.warning(
        "Infill split into %d islands: only the largest is filled (%.1f of %.1f mm2).",
        len(islands), islands[0].area, infill_boundary.area,
    )
    return islands[0]


def apply_global_pattern_to_layer(
        plan: GlobalPatternPlan,
        layer: Any,
        request: dict | PatternRequest | ResolvedPatternRequest | None = None,
        *,
        title: str = ApiDefaults.TITLE,
        fit_vertical_to_infill_boundary: bool = False,
        toolpath_cache: dict[bytes, _PatternPreviewResult] | None = None,
) -> _PatternPreviewResult | None:
    """
    Clip a previously generated global pattern to the largest contour in a layer.

    Holes are intentionally ignored for this first slicer-oriented pass.

    ``toolpath_cache`` (one dict per plan and request, see
    ``_build_layer_outputs``) reuses the toolpath of an earlier layer with the
    same wall offset. Everything after ``generate_wall_centerlines`` depends
    on the layer only through the walls and the infill boundary it returns,
    so the key is their WKB, not the raw contour: the slicer hands every layer
    of a prism the same outline with collinear vertices in different places
    -- (8000, 2955) on one layer, (8000, 2865) on the next, wherever the plane
    cut the STL's triangles -- and the offset is what removes them. MEASURED
    on honeycomb d90 / 8000x3000 mm / 33 layers of 1.5 mm: 33 distinct
    contours, one offset, ~18 s per layer otherwise recomputed 33 times. The
    reused result is the one the layer would compute itself (same function,
    same input), so determinism holds bit for bit.
    """
    resolved = plan.request if request is None else (
        request if isinstance(request, ResolvedPatternRequest) else resolve_pattern_request(request)
    )
    contour_points = _contour_points_from_layer(layer)
    if contour_points is None:
        return None

    final_contour = polygon_from_points(contour_points)
    wall_geometries, infill_boundary = generate_wall_centerlines(
        final_contour=final_contour,
        num_walls=resolved.num_walls,
        line_width=resolved.line_width,
        wall_spacing=resolved.wall_spacing,
        wall_join_style="round",
    )
    infill_boundary = _largest_infill_island(infill_boundary)

    cache_key = None
    if toolpath_cache is not None:
        cache_key = b"".join(
            [b"fit" if fit_vertical_to_infill_boundary else b"plan", title.encode()]
            + [shapely.to_wkb(geometry) for geometry in [infill_boundary, *wall_geometries]]
        )
        cached = toolpath_cache.get(cache_key)
        if cached is not None:
            return _copy_preview_for_layer(
                cached,
                layer,
                contour_points=np.asarray(contour_points, dtype=float),
                final_contour=final_contour,
            )

    if fit_vertical_to_infill_boundary:
        raw_columns = _build_boundary_fitted_vertical_columns(
            infill_boundary,
            resolved.line_width,
        )
        # ``infill_boundary`` is the next virtual wall centerline.  Do not
        # erode it by another half linewidth for these fitted centerlines.
        clipping_line_width = 0.0
    else:
        raw_columns = plan.raw_columns
        # ``infill_boundary`` is already the next virtual wall centerline.
        # Passing the physical linewidth here would make
        # ``clip_and_reconnect_columns`` erode it by another linewidth / 2,
        # creating an unintended perimeter gap and discarding valid partial
        # edge columns.
        clipping_line_width = 0.0

    columns = clip_and_reconnect_columns(
        raw_columns,
        infill_boundary,
        allow_boundary_reconnect=True,
        line_width=clipping_line_width,
    )
    _, _, infill_path_geometry, infill_start, infill_ordered = assemble_eulerian_path(
        infill_boundary=infill_boundary,
        columns=columns,
        line_width=resolved.line_width,
        wall_spacing=resolved.wall_spacing,
    )

    wall_gap_length = float(resolved.line_width / 2.0 + WallSettings.CONNECTION_GAP_EXTRA)
    full_path, full_path_geometry, start_point, end_point, fillet_protected_indices = _build_full_toolpath(
        infill_path_points=infill_ordered,
        infill_path_geometry=infill_path_geometry,
        wall_geometries=wall_geometries,
        wall_gap_length=wall_gap_length,
        wall_connection_angle_degrees=WallSettings.CONNECTION_ANGLE_DEGREES,
    )

    wall_centerlines = []
    for wall_geometry in wall_geometries:
        wall_line = _primary_boundary_line(wall_geometry)
        wall_centerlines.append(np.asarray(wall_line.coords, dtype=float))

    mesh_preview_points = apply_min_corner_radius_to_path(
        full_path,
        min_corner_radius=resolved.fillet_radius,
        min_turn_degrees=FilletSettings.MIN_TURN_DEGREES,
        segments_per_quadrant=FilletSettings.QUAD_SEGS,
        protected_indices=fillet_protected_indices,
    )
    mesh_preview_geometry: base.BaseGeometry | None
    if len(mesh_preview_points) >= 2:
        mesh_preview_geometry = LineString(mesh_preview_points)
    else:
        mesh_preview_geometry = None

    preview = _PatternPreviewResult(
        contour_points=np.asarray(contour_points, dtype=float),
        path_points=full_path,
        wall_centerlines=wall_centerlines,
        final_contour=final_contour,
        wall_geometries=wall_geometries,
        infill_boundary=infill_boundary,
        infill_path_points=np.asarray(infill_ordered, dtype=float),
        infill_path_geometry=infill_path_geometry,
        path_geometry=full_path_geometry,
        mesh_preview_geometry=mesh_preview_geometry,
        fillet_protected_indices=fillet_protected_indices,
        markers={
            "infill_start": infill_start,
            "path_start": start_point,
            "path_end": end_point,
        },
        title=title,
        unit=ApiDefaults.UNIT,
        num_walls=resolved.num_walls,
        line_width=resolved.line_width,
        wall_spacing=resolved.wall_spacing,
        fillet_radius=resolved.fillet_radius,
        infill_density=resolved.infill_density,
        infill_pattern=resolved.infill_pattern,
        extrude_height=resolved.extrude_height,
        layer_height=resolved.layer_height,
        bottom_layers=resolved.bottom_layers,
        top_layers=resolved.top_layers,
        line_distance=resolved.line_distance,
        estimated_density=float(resolved.estimated_density),
        pattern_distance_factor=float(resolved.pattern_distance_factor),
        pattern_amplitude=float(resolved.pattern_amplitude),
        pattern_step=float(resolved.pattern_step),
        infill_spacing=float(resolved.infill_spacing),
        z=float(_get_field(layer, "z", 0.0)),
    )
    if cache_key is not None:
        # A private copy: callers set fields such as mesh_kind on what they
        # get back, and that must not leak into the layers served later.
        toolpath_cache[cache_key] = _copy_preview_for_layer(preview, layer)
    return preview


def save_pattern_csv(
        preview_result: _PatternPreviewResult,
        *,
        filename: str | None = None,
) -> str:
    """Write the unrounded 2D toolpath points to CSV and return the path."""
    os.makedirs(PathSettings.RESULTS_DIR, exist_ok=True)
    base_name = _sanitize_name(preview_result.title)
    target_name = filename or f"{base_name}_path.csv"
    csv_path = os.path.join(PathSettings.RESULTS_DIR, os.path.basename(target_name))
    np.savetxt(csv_path, preview_result.path_points, delimiter=",", header="x,y", comments="")
    return csv_path


def _toolpath_points_for_robot(
        preview_result: _PatternPreviewResult,
        fillet_applied: bool,
) -> np.ndarray:
    """Return the centerline the robot follows for one layer.

    Ordinary infill paths use the rounded centerline. Solid layers reach this
    function as CSV-only vertical replacements created by
    ``_toolpath_previews_for_csv``; their original full-area STL preview remains
    separate and untouched.
    """
    if (
        fillet_applied
        and preview_result.mesh_kind != "solid"
        and preview_result.mesh_preview_geometry is not None
        and not preview_result.mesh_preview_geometry.is_empty
    ):
        return np.asarray(preview_result.mesh_preview_geometry.coords, dtype=float)
    return np.asarray(preview_result.path_points, dtype=float)


def _csv_layer_from_preview(preview_result: _PatternPreviewResult) -> dict[str, Any]:
    """Rebuild the minimal slicer layer needed for a CSV-only toolpath."""
    contour_points = np.asarray(preview_result.contour_points, dtype=float)
    if contour_points.ndim != 2 or contour_points.shape[1] != 2 or len(contour_points) < 3:
        raise ValueError(
            "Solid-layer CSV conversion requires at least three 2D contour points."
        )
    return {
        "z": float(
            preview_result.z
            if preview_result.z is not None
            else preview_result.layer_height / 2.0
        ),
        "method": "solid_layer_csv_toolpath",
        "contours": [{
            "id": 0,
            "area": float(abs(polygon_from_points(contour_points).area)),
            "is_largest": True,
            "exterior": contour_points.tolist(),
            "holes": [],
        }],
        "open_lines": [],
    }


def _solid_layer_csv_request(preview_result: _PatternPreviewResult) -> dict[str, Any]:
    """Return the manufacturing recipe used only for solid-layer CSV rows."""
    return {
        "num_walls": int(preview_result.num_walls),
        "line_width": float(preview_result.line_width),
        # Solid CSV rasters always close the gap between the innermost wall and
        # the first/last infill bead.  This does not alter the solid STL
        # footprint because the replacement preview is CSV-only.
        "wall_spacing": 0.0,
        "fillet_radius": float(preview_result.fillet_radius),
        "infill_density": 100.0,
        "infill_pattern": "vertical",
        "extrude_height": float(preview_result.extrude_height),
        "layer_height": float(preview_result.layer_height),
        # These temporary previews represent printable centerlines.  Keeping
        # them out of sandwich mode is what prevents the CSV conversion from
        # changing the solid footprint consumed by the STL exporter.
        "bottom_layers": 0,
        "top_layers": 0,
    }


def _toolpath_previews_for_csv(
        preview_results: list[_PatternPreviewResult],
) -> list[_PatternPreviewResult]:
    """Replace solid layers with temporary vertical, fully dense toolpaths.

    The input previews are never mutated.  Infill previews are returned as-is,
    while every ``mesh_kind == "solid"`` preview is regenerated solely for
    robot CSV export with a vertical pattern at 100 % density.  STL generation
    therefore continues to see the original solid previews and their full-area
    footprints.
    """
    ordered = sorted(preview_results, key=lambda preview: float(preview.z or 0.0))
    solid_previews = [preview for preview in ordered if preview.mesh_kind == "solid"]
    if not solid_previews:
        return ordered

    layers = [_csv_layer_from_preview(preview) for preview in ordered]
    contour_arrays = [
        np.asarray(preview.contour_points, dtype=float)
        for preview in ordered
    ]
    minx = min(float(points[:, 0].min()) for points in contour_arrays)
    miny = min(float(points[:, 1].min()) for points in contour_arrays)
    maxx = max(float(points[:, 0].max()) for points in contour_arrays)
    maxy = max(float(points[:, 1].max()) for points in contour_arrays)
    z_min = min(
        float(layer["z"]) - float(preview.layer_height) / 2.0
        for layer, preview in zip(layers, ordered)
    )
    z_max = max(
        float(layer["z"]) + float(preview.layer_height) / 2.0
        for layer, preview in zip(layers, ordered)
    )
    slicer_output = {
        "layer_height": float(solid_previews[0].layer_height),
        "model_bounds": {
            "min": [minx, miny, z_min],
            "max": [maxx, maxy, z_max],
            "size": [maxx - minx, maxy - miny, z_max - z_min],
        },
        "layers": layers,
        "total_layers": len(layers),
        "closed_layers": len(layers),
        "failed_layers": 0,
        "noise_layers": 0,
    }
    solid_plan = build_global_pattern_plan(
        _solid_layer_csv_request(solid_previews[0]),
        slicer_output,
    )

    csv_previews: list[_PatternPreviewResult] = []
    for original, layer in zip(ordered, layers):
        if original.mesh_kind != "solid":
            csv_previews.append(original)
            continue
        replacement = apply_global_pattern_to_layer(
            solid_plan,
            layer,
            title=original.title,
            fit_vertical_to_infill_boundary=True,
        )
        if replacement is None:
            raise ValueError("Solid layer produced no printable CSV toolpath.")
        # Preserve the original layer role on the CSV-only replacement.  The
        # regenerated preview deliberately uses a regular (non-sandwich)
        # pattern request, so without this assignment its dataclass default
        # would incorrectly describe the layer as ordinary infill.
        replacement.mesh_kind = "solid"
        csv_previews.append(replacement)
    return csv_previews


#: Column order of the robot toolpath CSV.
TOOLPATH_CSV_COLUMNS = (
    "path_id",
    "point_index",
    "layer",
    "x",
    "y",
    "z",
    "print_type",
    "feature_type",
    "path_type",
)

#: Process tag carried by every toolpath row.
TOOLPATH_PRINT_TYPE = "resin"
TOOLPATH_FEATURE_TYPE = "flow"


def _polyline_cumulative_lengths(points: np.ndarray) -> np.ndarray:
    """Return cumulative distance at every point of a 2D polyline."""
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return np.zeros(0, dtype=float)
    if len(pts) == 1:
        return np.zeros(1, dtype=float)
    return np.concatenate((
        np.zeros(1, dtype=float),
        np.cumsum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)),
    ))


def _toolpath_split_index(
        preview_result: _PatternPreviewResult,
        exported_points: np.ndarray,
) -> int | None:
    """Locate the preserved infill-to-wall seam in the exported centerline."""
    protected = list(getattr(preview_result, "fillet_protected_indices", []))
    raw_points = np.asarray(preview_result.path_points, dtype=float)
    if not protected or len(raw_points) == 0 or len(exported_points) == 0:
        return None

    raw_index = int(protected[0])
    if not 0 <= raw_index < len(raw_points):
        return None
    seam = raw_points[raw_index]
    tolerance = max(GeometrySettings.COORD_EPSILON * 10.0, 1e-9)
    candidates = np.flatnonzero(
        np.linalg.norm(np.asarray(exported_points, dtype=float) - seam, axis=1) <= tolerance
    )
    if len(candidates) == 0:
        return None
    if len(candidates) == 1:
        return int(candidates[0])

    # A wall seam can revisit an XY coordinate.  Pick the occurrence at the
    # same relative arc-length position as the protected raw-path vertex.
    raw_lengths = _polyline_cumulative_lengths(raw_points)
    exported_lengths = _polyline_cumulative_lengths(exported_points)
    raw_ratio = raw_lengths[raw_index] / raw_lengths[-1] if raw_lengths[-1] > 0.0 else 0.0
    exported_total = exported_lengths[-1]
    return int(min(
        candidates,
        key=lambda index: abs(
            (exported_lengths[int(index)] / exported_total if exported_total > 0.0 else 0.0)
            - raw_ratio
        ),
    ))


def _toolpath_points_and_types_for_robot(
        preview_result: _PatternPreviewResult,
        fillet_applied: bool,
) -> tuple[np.ndarray, list[str]]:
    """Return one continuous flow path plus solid/infill/wall point tags.

    ``feature_type=flow`` and the single per-layer ``path_id`` identify one
    continuous deposition per layer.  ``path_type`` carries the richer geometry
    role without splitting or duplicating the centerline.
    """
    points = _toolpath_points_for_robot(preview_result, fillet_applied)
    if points.size == 0:
        return points, []
    if preview_result.mesh_kind == "solid":
        return points, ["solid"] * len(points)

    wall_centerlines = list(getattr(preview_result, "wall_centerlines", []))
    infill_points = np.asarray(
        getattr(preview_result, "infill_path_points", np.zeros((0, 2))),
        dtype=float,
    )
    if not wall_centerlines:
        return points, ["infill"] * len(points)
    if len(infill_points) < 2:
        return points, ["wall"] * len(points)

    split_index = _toolpath_split_index(preview_result, points)
    if split_index is None:
        raise ValueError("Could not locate the infill-to-wall seam for toolpath CSV export.")
    return points, ["infill"] * split_index + ["wall"] * (len(points) - split_index)


def save_toolpath_csv(
        preview_results: list[_PatternPreviewResult],
        *,
        filename: str | None = None,
        output_dir: str | None = None,
) -> str:
    """Write the whole stack's robot toolpath as one CSV and return its path.

    Columns are ``TOOLPATH_CSV_COLUMNS``, ordered by z and then by point.
    Layers are numbered from 1 so the tags read like the printed part.
    """
    if not preview_results:
        raise ValueError("No layer previews were provided.")

    results_dir = _results_dir(output_dir)
    base_name = _sanitize_name(preview_results[0].title)
    target_name = filename or f"{base_name}_toolpath.csv"
    csv_path = os.path.join(results_dir, os.path.basename(target_name))

    # TODO: Add seam positioning and a common end/start point between
    # consecutive layers instead of relying on downstream endpoint connectors.
    ordered = _toolpath_previews_for_csv(preview_results)
    rows: list[tuple] = []
    for layer, preview in enumerate(ordered, start=1):
        fillet_applied = resolve_fillet(preview.fillet_radius, preview.line_width).applied
        z_center = float(preview.z if preview.z is not None else preview.layer_height / 2.0)
        points, path_types = _toolpath_points_and_types_for_robot(preview, fillet_applied)
        path_id = f"L{layer:03d}_{TOOLPATH_FEATURE_TYPE}_path"
        for point_index, ((x, y), path_type) in enumerate(zip(points, path_types)):
            rows.append((
                path_id,
                point_index,
                layer,
                f"{float(x):.6f}",
                f"{float(y):.6f}",
                f"{z_center:.6f}",
                TOOLPATH_PRINT_TYPE,
                TOOLPATH_FEATURE_TYPE,
                path_type,
            ))

    if not rows:
        raise ValueError("The generated toolpath is empty; nothing to export as CSV.")

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(TOOLPATH_CSV_COLUMNS)
        writer.writerows(rows)
    return csv_path


def save_pattern_csv_rounded(
        preview_result: _PatternPreviewResult,
        *,
        filename: str | None = None,
) -> str:
    """Write the rounded/filleted preview path to CSV and return the path."""
    if preview_result.mesh_preview_geometry is None or preview_result.mesh_preview_geometry.is_empty:
        raise ValueError("No rounded preview path is available for CSV export.")

    os.makedirs(PathSettings.RESULTS_DIR, exist_ok=True)
    base_name = _sanitize_name(preview_result.title)
    target_name = filename or f"{base_name}_path_filleted.csv"
    csv_path = os.path.join(PathSettings.RESULTS_DIR, os.path.basename(target_name))
    rounded_points = np.asarray(preview_result.mesh_preview_geometry.coords, dtype=float)
    np.savetxt(csv_path, rounded_points, delimiter=",", header="x,y", comments="")
    return csv_path


def _build_preview_mesh(
        preview_result: _PatternPreviewResult,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a single-preview mesh using either plain infill or sandwich layers."""
    if preview_result.extrude_height <= 0.0:
        raise ValueError("'extrude_height' must be > 0 to generate 3D outputs.")

    mesh_radius = (
        preview_result.fillet_radius
        if preview_result.fillet_radius > GeometrySettings.AREA_EPSILON
        else 0.0
    )
    mesh_line_width = _mesh_export_line_width(preview_result.line_width)
    if preview_result.bottom_layers > 0 or preview_result.top_layers > 0:
        solid_footprint_geometry = None
        if preview_result.wall_centerlines:
            solid_footprint_geometry = build_solid_geometry_from_centerline_contour(
                preview_result.wall_centerlines[0],
                thickness=mesh_line_width,
                fillet_radius=mesh_radius,
            )
        return generate_sandwich_mesh(
            contour_2d=preview_result.contour_points,
            infill_path_2d=preview_result.path_points,
            total_height=preview_result.extrude_height,
            thickness=mesh_line_width,
            fillet_radius=mesh_radius,
            bottom_layers=preview_result.bottom_layers,
            top_layers=preview_result.top_layers,
            layer_height=preview_result.layer_height,
            fillet_protected_indices=preview_result.fillet_protected_indices,
            solid_footprint_geometry=solid_footprint_geometry,
        )

    return generate_mesh(
        preview_result.path_points,
        height=preview_result.extrude_height,
        thickness=mesh_line_width,
        mesh_min_corner_radius=mesh_radius,
        fillet_protected_indices=preview_result.fillet_protected_indices,
    )


def _mesh_export_line_width(line_width: float) -> float:
    extra = max(0.0, float(MeshSettings.LINE_WIDTH_EXPORT_EXTRA))
    return float(line_width) + extra


def _solid_footprint_geometry_from_preview(preview_result: _PatternPreviewResult):
    mesh_radius = (
        preview_result.fillet_radius
        if preview_result.fillet_radius > GeometrySettings.AREA_EPSILON
        else 0.0
    )
    mesh_line_width = _mesh_export_line_width(preview_result.line_width)
    if preview_result.wall_centerlines:
        return build_solid_geometry_from_centerline_contour(
            preview_result.wall_centerlines[0],
            thickness=mesh_line_width,
            fillet_radius=mesh_radius,
        )
    return build_solid_contour_geometry(
        preview_result.contour_points,
        fillet_radius=mesh_radius,
    )


def _mark_solid_layer_kinds(previews: list[_PatternPreviewResult]) -> None:
    if not previews:
        return

    bottom_count = max(0, int(previews[0].bottom_layers))
    top_count = max(0, int(previews[0].top_layers))
    top_start = max(0, len(previews) - top_count)

    for index, preview in enumerate(previews):
        preview.mesh_kind = "solid" if index < bottom_count or index >= top_start else "infill"


def _build_layer_mesh(
        preview_result: _PatternPreviewResult,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one physical layer at its slicer Z position."""
    layer_height = float(preview_result.layer_height)
    if layer_height <= GeometrySettings.AREA_EPSILON:
        raise ValueError("'layer_height' must be > 0 for multilayer mesh generation.")

    z_center = float(preview_result.z if preview_result.z is not None else layer_height / 2.0)
    z_base = z_center - layer_height / 2.0
    mesh_radius = (
        preview_result.fillet_radius
        if preview_result.fillet_radius > GeometrySettings.AREA_EPSILON
        else 0.0
    )

    if preview_result.mesh_kind == "solid":
        vertices, faces = _build_raw_solid_geometry_mesh(
            _solid_footprint_geometry_from_preview(preview_result),
            height=layer_height,
        )
    else:
        vertices, faces = _build_raw_path_mesh(
            preview_result.path_points,
            height=layer_height,
            thickness=_mesh_export_line_width(preview_result.line_width),
            mesh_min_corner_radius=mesh_radius,
            fillet_protected_indices=preview_result.fillet_protected_indices,
        )

    if len(vertices) == 0 or len(faces) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    return _translate_mesh_z(vertices, z_base), faces


def _build_multilayer_mesh(
        previews: list[_PatternPreviewResult],
) -> tuple[np.ndarray, np.ndarray]:
    """Combine all layer previews into the final STL mesh."""
    ordered_previews = sorted(previews, key=lambda preview: float(preview.z or 0.0))
    _mark_solid_layer_kinds(ordered_previews)

    mesh_parts = []
    for preview in ordered_previews:
        vertices, faces = _build_layer_mesh(preview)
        if len(vertices) > 0 and len(faces) > 0:
            mesh_parts.append((vertices, faces))

    vertices, faces = _concatenate_mesh_parts(mesh_parts)
    if len(vertices) == 0 or len(faces) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    return _finalize_mesh(vertices, faces, log_prefix="generate_multilayer_mesh")


def save_pattern_stl(
        preview_result: _PatternPreviewResult,
        *,
        filename: str | None = None,
) -> str:
    """Export one preview result to STL and return the written path."""
    vertices, faces = _build_preview_mesh(preview_result)
    os.makedirs(PathSettings.RESULTS_DIR, exist_ok=True)
    base_name = _sanitize_name(preview_result.title)
    target_name = filename or f"{base_name}.stl"
    stl_path = os.path.join(PathSettings.RESULTS_DIR, os.path.basename(target_name))
    export_stl(vertices, faces, stl_path)
    return stl_path


def save_pattern_stack_stl(
        preview_results: list[_PatternPreviewResult],
        *,
        filename: str | None = None,
        output_dir: str | None = None,
) -> str:
    """Export a multilayer preview stack to a single STL file.

    Every layer is one closed body; they are stacked, not fused, so the STL
    holds one volume per layer.
    """
    if not preview_results:
        raise ValueError("No layer previews are available for STL export.")

    vertices, faces = _build_multilayer_mesh(preview_results)
    results_dir = _results_dir(output_dir)
    base_name = _sanitize_name(preview_results[0].title)
    target_name = filename or f"{base_name}.stl"
    stl_path = os.path.join(results_dir, os.path.basename(target_name))
    export_stl(vertices, faces, stl_path)
    return stl_path


def save_pattern_stp(
        preview_result: _PatternPreviewResult,
        *,
        filename: str | None = None,
        stl_path: str | None = None,
) -> str:
    """Export one preview result to STEP through the stable STL conversion path."""
    os.makedirs(PathSettings.RESULTS_DIR, exist_ok=True)
    base_name = _sanitize_name(preview_result.title)
    target_name = filename or f"{base_name}.step"
    step_path = os.path.join(PathSettings.RESULTS_DIR, os.path.basename(target_name))
    source_stl = stl_path or save_pattern_stl(preview_result, filename=f"{Path(step_path).stem}.stl")
    stl_to_step(Path(source_stl), Path(step_path))
    return step_path


def save_pattern_stack_stp(
        preview_results: list[_PatternPreviewResult],
        *,
        filename: str | None = None,
        stl_path: str | None = None,
        output_dir: str | None = None,
) -> str:
    """Export a multilayer preview stack to STEP through the stable STL conversion path."""
    if not preview_results:
        raise ValueError("No layer previews are available for STEP export.")

    results_dir = _results_dir(output_dir)
    base_name = _sanitize_name(preview_results[0].title)
    target_name = filename or f"{base_name}.step"
    step_path = os.path.join(results_dir, os.path.basename(target_name))
    source_stl = stl_path or save_pattern_stack_stl(
        preview_results, filename=f"{Path(step_path).stem}.stl", output_dir=results_dir,
    )
    stl_to_step(Path(source_stl), Path(step_path))
    return step_path


def _build_public_output_from_previews(
        previews: list[_PatternPreviewResult],
        options: PatternOutputOptions,
) -> dict[str, Any]:
    first_preview = previews[0]
    output = {
        "path_points": first_preview.path_points,
        "wall_centerlines": first_preview.wall_centerlines,
        "toolpath_csv_path": None,
        "stl_path": None,
        "step_path": None,
    }
    if options.toolpath_csv_file:
        output["toolpath_csv_path"] = save_toolpath_csv(previews, output_dir=options.output_dir)
    # STEP is derived from the STL, so either flag needs the surface written.
    if options.stl_file or options.step_file:
        stl_path = save_pattern_stack_stl(previews, output_dir=options.output_dir)
        if options.stl_file:
            output["stl_path"] = stl_path
        if options.step_file:
            output["step_path"] = save_pattern_stack_stp(
                previews, stl_path=stl_path, output_dir=options.output_dir,
            )
    return output


def _build_pattern_preview(payload: dict | PatternRequest | ResolvedPatternRequest) -> _PatternPreviewResult:
    """
    Internal single-layer build result used by plotting and export helpers.

    Dict payloads with ``contour_points`` use the legacy compatibility wrapper.
    """
    if isinstance(payload, dict) and "contour_points" in payload:
        request = resolve_pattern_request(payload)
        slicer_output = _legacy_slicer_output_from_contour(
            payload["contour_points"],
            layer_height=request.layer_height,
        )
        plan = build_global_pattern_plan(request, slicer_output)
        layer = _layers_from_slicer_output(slicer_output)[0]
        preview = apply_global_pattern_to_layer(
            plan,
            layer,
            title=str(payload.get("title", ApiDefaults.TITLE)),
        )
        if preview is None:
            raise ValueError("Legacy contour produced no printable layer.")
        return preview

    raise ValueError("_build_pattern_preview requires a legacy payload with 'contour_points'.")


def _layer_output(preview_result: _PatternPreviewResult) -> dict[str, Any]:
    return {
        "z": preview_result.z,
        "mesh_kind": preview_result.mesh_kind,
        "path_points": preview_result.path_points,
        "wall_centerlines": preview_result.wall_centerlines,
        "markers": preview_result.markers,
    }


def _build_layer_outputs(
        plan: GlobalPatternPlan,
        slicer_output: Any,
        *,
        title: str,
) -> list[_PatternPreviewResult]:
    """One preview per slicer layer; layers that offset to the same walls
    share one toolpath computation (see ``apply_global_pattern_to_layer``)."""
    previews: list[_PatternPreviewResult] = []
    toolpath_cache: dict[bytes, _PatternPreviewResult] = {}
    for layer in _sorted_layers_from_slicer_output(slicer_output):
        preview = apply_global_pattern_to_layer(
            plan, layer, title=title, toolpath_cache=toolpath_cache,
        )
        if preview is not None:
            previews.append(preview)
    _mark_solid_layer_kinds(previews)
    return previews


def _copy_preview_for_layer(
        preview: _PatternPreviewResult,
        layer: Any,
        **layer_fields: Any,
) -> _PatternPreviewResult:
    """A cached layer's result for another layer: its z and ``layer_fields``.

    Arrays, lists and dicts are copied, so no caller can change one layer's
    path through another's; shapely geometries are immutable and stay shared.
    """
    copied = copy.copy(preview)
    for preview_field in fields(copied):
        value = getattr(copied, preview_field.name)
        if isinstance(value, (np.ndarray, list, dict)):
            setattr(copied, preview_field.name, copy.deepcopy(value))
    for name, value in layer_fields.items():
        setattr(copied, name, value)
    copied.z = float(_get_field(layer, "z", 0.0))
    return copied


def generate_pattern(
        payload: dict | PatternRequest,
        slicer_output: Any | None = None,
        options: dict | PatternOutputOptions | None = None,
) -> dict[str, Any]:
    """
    Generate a pattern from either legacy or slicer-oriented inputs.

    Public output is intentionally compact and stable.
    Plotting and export auxiliary state lives in internal helpers only.
    """
    if isinstance(payload, dict) and "request" in payload:
        request_payload = payload["request"]
        slicer_output = payload.get("slicer_output", slicer_output)
        options = payload.get("options", options)
    else:
        request_payload = payload

    output_options = _coerce_output_options(options if options is not None else (
        payload if isinstance(payload, dict) else {}
    ))
    request = resolve_pattern_request(request_payload)

    if slicer_output is None:
        if not isinstance(payload, dict) or "contour_points" not in payload:
            raise ValueError("generate_pattern requires 'slicer_output' or legacy 'contour_points'.")
        slicer_output = _legacy_slicer_output_from_contour(
            payload["contour_points"],
            layer_height=request.layer_height,
        )

    plan = build_global_pattern_plan(request, slicer_output)
    previews = _build_layer_outputs(plan, slicer_output, title=output_options.title)
    if not previews:
        return {
            "path_points": np.zeros((0, 2), dtype=float),
            "wall_centerlines": [],
            "layers": [],
            "toolpath_csv_path": None,
            "stl_path": None,
            "step_path": None,
        }

    output = _build_public_output_from_previews(previews, output_options)
    output["layers"] = [_layer_output(preview) for preview in previews]
    return output

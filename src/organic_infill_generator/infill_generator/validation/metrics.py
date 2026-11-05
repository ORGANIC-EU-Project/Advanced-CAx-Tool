"""Numeric validation metrics for generated or imported infill paths."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable, Mapping, Sequence

import numpy as np
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point, base
from shapely.ops import unary_union
from shapely.strtree import STRtree

from ..settings import GeometrySettings


def _as_points(path_points: object) -> np.ndarray:
    points = np.asarray(path_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        return np.zeros((0, 2), dtype=float)
    return points


def _as_polylines(path_points: object) -> list[np.ndarray]:
    if path_points is None:
        return []
    if isinstance(path_points, LineString):
        return [_as_points(path_points.coords)]
    if isinstance(path_points, MultiLineString):
        return [_as_points(line.coords) for line in path_points.geoms if len(line.coords) >= 2]
    if isinstance(path_points, (list, tuple)) and path_points:
        first = path_points[0]
        if isinstance(first, LineString):
            return [_as_points(line.coords) for line in path_points if len(line.coords) >= 2]
        if isinstance(first, (list, tuple, np.ndarray)):
            arr = np.asarray(first, dtype=float)
            if arr.ndim == 2:
                return [_as_points(item) for item in path_points if len(_as_points(item)) >= 2]
    points = _as_points(path_points)
    return [points] if len(points) >= 2 else []


def _line_from_polylines(path_points: object) -> base.BaseGeometry | None:
    lines = [LineString(polyline) for polyline in _as_polylines(path_points) if len(polyline) >= 2]
    if not lines:
        return None
    if len(lines) == 1:
        return lines[0]
    merged = unary_union(lines)
    return merged if not merged.is_empty else MultiLineString(lines)


def _safe_area(geometry: base.BaseGeometry | None) -> float:
    if geometry is None or geometry.is_empty:
        return 0.0
    return float(getattr(geometry, "area", 0.0))


def _safe_length(geometry: base.BaseGeometry | None) -> float:
    if geometry is None or geometry.is_empty:
        return 0.0
    return float(getattr(geometry, "length", 0.0))


def _effective_printable_geometry(
        printable_geometry: base.BaseGeometry | None,
        holes: Iterable[base.BaseGeometry] = (),
) -> base.BaseGeometry | None:
    if printable_geometry is None or printable_geometry.is_empty:
        return printable_geometry
    effective = printable_geometry
    for hole in holes:
        if hole is not None and not hole.is_empty:
            effective = effective.difference(hole)
    return effective.buffer(0) if effective is not None and not effective.is_valid else effective


def swept_geometry(
        path_points: object,
        line_width: float,
) -> base.BaseGeometry | None:
    """Return the 2-D material footprint implied by centerlines and line width."""
    line = _line_from_polylines(path_points)
    if line is None or line.is_empty or line_width <= GeometrySettings.AREA_EPSILON:
        return None
    swept = line.buffer(
        float(line_width) / 2.0,
        cap_style=1,
        join_style=1,
    )
    if not swept.is_valid:
        repaired = swept.buffer(0)
        if not repaired.is_empty:
            swept = repaired
    return swept if not swept.is_empty else None


def _line_length_inside(
        path_points: object,
        region: base.BaseGeometry | None,
) -> float:
    line = _line_from_polylines(path_points)
    if line is None or line.is_empty:
        return 0.0
    if region is None or region.is_empty:
        return _safe_length(line)
    return _safe_length(line.intersection(region))


def coverage_metrics(
        *,
        path_points: object,
        line_width: float,
        printable_geometry: base.BaseGeometry | None,
) -> dict[str, float]:
    """
    Measure infill density against the effective printable domain.

    The primary density is linear: centerline length times line width divided
    by domain area. Swept density is reported separately to expose overlap loss
    and saturation effects.
    """
    swept = swept_geometry(path_points, line_width)
    printable_area = _safe_area(printable_geometry)
    centerline_length = _line_length_inside(path_points, printable_geometry)
    linear_area = centerline_length * max(float(line_width), 0.0)
    swept_area = _safe_area(swept)

    if swept is None or printable_geometry is None or printable_geometry.is_empty:
        covered_area = 0.0
        outside_area = swept_area
    else:
        covered_area = _safe_area(swept.intersection(printable_geometry))
        outside_area = _safe_area(swept.difference(printable_geometry))

    linear_density = (
        100.0 * linear_area / printable_area
        if printable_area > GeometrySettings.AREA_EPSILON
        else 0.0
    )
    swept_density = (
        100.0 * covered_area / printable_area
        if printable_area > GeometrySettings.AREA_EPSILON
        else 0.0
    )
    return {
        "printable_area": printable_area,
        "infill_centerline_length": centerline_length,
        "linear_material_area": linear_area,
        "linear_density_percent": linear_density,
        "swept_area": swept_area,
        "covered_area": covered_area,
        "covered_density_percent": swept_density,
        "swept_density_percent": swept_density,
        "density_overlap_loss_percent": linear_density - swept_density,
        "outside_area": outside_area,
        "outside_area_percent": (
            100.0 * outside_area / max(swept_area, GeometrySettings.AREA_EPSILON)
            if swept_area > GeometrySettings.AREA_EPSILON
            else 0.0
        ),
    }


def regional_coverage_metrics(
        *,
        path_points: object,
        line_width: float,
        printable_geometry: base.BaseGeometry | None,
        region_geometry: base.BaseGeometry | None,
        prefix: str,
) -> dict[str, float]:
    """Measure linear and swept density inside a selected printable sub-region."""
    if (
        printable_geometry is None
        or printable_geometry.is_empty
        or region_geometry is None
        or region_geometry.is_empty
    ):
        return {
            f"{prefix}_area": 0.0,
            f"{prefix}_centerline_length": 0.0,
            f"{prefix}_linear_density_percent": 0.0,
            f"{prefix}_covered_area": 0.0,
            f"{prefix}_covered_density_percent": 0.0,
        }

    effective_region = printable_geometry.intersection(region_geometry)
    if effective_region.is_empty:
        return {
            f"{prefix}_area": 0.0,
            f"{prefix}_centerline_length": 0.0,
            f"{prefix}_linear_density_percent": 0.0,
            f"{prefix}_covered_area": 0.0,
            f"{prefix}_covered_density_percent": 0.0,
        }

    swept = swept_geometry(path_points, line_width)
    covered_area = _safe_area(swept.intersection(effective_region)) if swept is not None else 0.0
    region_area = _safe_area(effective_region)
    centerline_length = _line_length_inside(path_points, effective_region)
    linear_density = (
        100.0 * centerline_length * max(float(line_width), 0.0) / region_area
        if region_area > GeometrySettings.AREA_EPSILON
        else 0.0
    )
    return {
        f"{prefix}_area": region_area,
        f"{prefix}_centerline_length": centerline_length,
        f"{prefix}_linear_density_percent": linear_density,
        f"{prefix}_covered_area": covered_area,
        f"{prefix}_covered_density_percent": (
            100.0 * covered_area / region_area
            if region_area > GeometrySettings.AREA_EPSILON
            else 0.0
        ),
    }


def _node_key(point: Sequence[float], tolerance: float) -> tuple[int, int]:
    scale = 1.0 / max(float(tolerance), GeometrySettings.COORD_EPSILON)
    return int(round(float(point[0]) * scale)), int(round(float(point[1]) * scale))


def _segments(path_points: object) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for polyline in _as_polylines(path_points):
        for start, end in zip(polyline[:-1], polyline[1:]):
            if float(np.linalg.norm(end - start)) > GeometrySettings.LENGTH_EPSILON:
                result.append(np.array([start, end], dtype=float))
    return result


def _count_crossings_and_overlaps(segments: list[np.ndarray], tolerance: float) -> tuple[int, int]:
    """Exact counts of self-crossings and collinear overlaps.

    An STRtree keeps this O(n log n), so a real toolpath of ~1500 segments
    counts in well under a second.
    """
    if not segments:
        return 0, 0

    lines = [LineString(segment) for segment in segments]
    tree = STRtree(lines)
    crossing_count = 0
    overlap_count = 0
    for i, first in enumerate(lines):
        for raw in tree.query(first):
            j = int(raw)
            # Only look forward, and skip the neighbour that legitimately
            # shares an endpoint with this segment.
            if j <= i + 1:
                continue
            second = lines[j]
            intersection = first.intersection(second)
            if intersection.is_empty:
                continue
            if intersection.geom_type == "Point":
                point = Point(intersection)
                endpoints = (
                    Point(first.coords[0]), Point(first.coords[-1]),
                    Point(second.coords[0]), Point(second.coords[-1]),
                )
                if any(point.distance(endpoint) <= tolerance for endpoint in endpoints):
                    continue
                crossing_count += 1
            elif intersection.geom_type in {"LineString", "MultiLineString"}:
                if _safe_length(intersection) > tolerance:
                    overlap_count += 1
            else:
                crossing_count += 1
    return crossing_count, overlap_count


def _wall_geometries(wall_centerlines: object) -> list[base.BaseGeometry]:
    """Accept walls as point arrays or as shapely geometries.

    Callers differ: ``preview_metrics`` holds shapely ``wall_geometries`` while
    ``generate_pattern`` returns ``wall_centerlines`` as ``(N, 2)`` arrays. An
    area geometry is reduced to its boundary, since the reference for a
    clearance is the wall outline, not its interior.
    """
    result: list[base.BaseGeometry] = []
    for wall in (wall_centerlines or []):
        if isinstance(wall, base.BaseGeometry):
            geometry = wall.boundary if wall.geom_type in {"Polygon", "MultiPolygon"} else wall
            if geometry is not None and not geometry.is_empty:
                result.append(geometry)
            continue
        points = np.asarray(wall, dtype=float)
        if points.ndim == 2 and len(points) >= 2:
            result.append(LineString(points))
    return result


def wall_clearance_metrics(
        *,
        path_points: object,
        wall_centerlines: object,
        line_width: float,
) -> dict[str, float | int]:
    """How close the infill runs to the wall, and how much material that doubles.

    There was no metric for this, and it is the one that catches what happens on
    oblique and curved contours: the nominal infill-to-wall clearance measured a
    consistent 3.70 mm on every axis-aligned outline, but collapsed to 2.62 mm on
    an octagon and 1.39 mm on an ellipse -- roughly a third of nominal, with
    beads then overlapping the wall. Some overlap is intended, so that the infill
    bonds to the perimeter; the signal is the *departure from nominal*, not the
    overlap itself.

    ``min_wall_clearance_mm`` is the worst approach anywhere on the path.
    ``infill_below_half_width_mm`` is how much path length sits closer than half
    a line width, which is where doubling starts to be real.
    """
    empty = {
        "min_wall_clearance_mm": 0.0,
        "infill_below_line_width_mm": 0.0,
        "infill_below_half_width_mm": 0.0,
        "wall_overlap_area_mm2": 0.0,
        "infill_length_mm": 0.0,
    }
    walls = _wall_geometries(wall_centerlines)
    segments = _segments(path_points)
    if not walls or not segments:
        return empty

    wall = unary_union(walls)
    half = max(float(line_width), GeometrySettings.COORD_EPSILON) / 2.0
    # Segments lying on the wall itself are the wall, not infill approaching it.
    on_wall = 0.05 * float(line_width)
    infill = [LineString(s) for s in segments if LineString(s).distance(wall) > on_wall]
    if not infill:
        return empty

    minimum = float("inf")
    below_full = 0.0
    below_half = 0.0
    total = 0.0
    touching: list[base.BaseGeometry] = []
    for seg in infill:
        distance = seg.distance(wall)
        minimum = min(minimum, distance)
        total += seg.length
        if distance < float(line_width):
            below_full += seg.length
            # Two half-width beads can only overlap below one full line width,
            # so everything further away contributes nothing to the area and
            # does not need buffering. That keeps this off the O(n) buffer path:
            # ~1400 segments drop to a few dozen.
            touching.append(seg)
        if distance < half:
            below_half += seg.length

    doubled = (unary_union([s.buffer(half) for s in touching]).intersection(wall.buffer(half))
               if touching else GeometryCollection())
    return {
        "min_wall_clearance_mm": float(minimum),
        "infill_below_line_width_mm": float(below_full),
        "infill_below_half_width_mm": float(below_half),
        "wall_overlap_area_mm2": float(_safe_area(doubled)),
        "infill_length_mm": float(total),
    }


def eulerian_metrics(
        path_points: object,
        *,
        tolerance: float = 1e-6,
) -> dict[str, int | bool]:
    """Noderize the segment network, then evaluate eulerian trail feasibility."""
    segments = _segments(path_points)
    if not segments:
        return {
            "graph_nodes": 0,
            "graph_edges": 0,
            "graph_components": 0,
            "odd_degree_nodes": 0,
            "is_eulerian_path_candidate": False,
        }

    # unary_union nodes the linework: every crossing becomes a vertex of both lines.
    noded = unary_union([LineString(segment) for segment in segments])
    lines = list(noded.geoms) if hasattr(noded, "geoms") else [noded]
    grid = max(float(tolerance), 1e-8)
    adjacency: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    for line in lines:
        nodes = [_node_key(coord, grid) for coord in line.coords]
        for first, second in zip(nodes[:-1], nodes[1:]):
            if first != second:
                adjacency[first].add(second)
                adjacency[second].add(first)

    node_count = len(adjacency)
    edge_count = sum(len(neighbors) for neighbors in adjacency.values()) // 2
    remaining = set(adjacency)
    components = 0
    while remaining:
        components += 1
        queue: deque[tuple[int, int]] = deque([remaining.pop()])
        while queue:
            for neighbor in adjacency[queue.popleft()]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
    odd_nodes = sum(1 for neighbors in adjacency.values() if len(neighbors) % 2 == 1)
    return {
        "graph_nodes": int(node_count),
        "graph_edges": int(edge_count),
        "graph_components": int(components),
        "odd_degree_nodes": int(odd_nodes),
        "is_eulerian_path_candidate": components == 1 and odd_nodes in {0, 2},
    }


def ordered_path_continuity_metrics(
        path_points: object,
        *,
        tolerance: float,
) -> dict[str, float | int | bool]:
    """Check whether extrusion polylines form one continuous ordered robot path."""
    polylines = _as_polylines(path_points)
    if not polylines:
        return {
            "ordered_gap_count": 0,
            "jump_count": 0,
            "max_ordered_gap": 0.0,
            "is_single_path": False,
            "is_ordered_continuous": False,
        }
    gaps = []
    for before, after in zip(polylines[:-1], polylines[1:]):
        gaps.append(float(np.linalg.norm(after[0] - before[-1])))
    jump_count = sum(1 for gap in gaps if gap > tolerance)
    return {
        "ordered_gap_count": int(jump_count),
        "jump_count": int(jump_count),
        "max_ordered_gap": max(gaps, default=0.0),
        "is_single_path": len(polylines) == 1 or jump_count == 0,
        "is_ordered_continuous": len(polylines) == 1 or jump_count == 0,
    }


def path_quality_metrics(
        *,
        path_points: object,
        printable_geometry: base.BaseGeometry | None,
        line_width: float,
        long_segment_factor: float = 8.0,
) -> dict[str, float | int | bool]:
    """Measure path continuity, intersections, and suspicious outside segments."""
    polylines = _as_polylines(path_points)
    line = _line_from_polylines(polylines)
    segments = _segments(polylines)
    segment_lengths = [float(LineString(segment).length) for segment in segments]
    max_segment = max(segment_lengths, default=0.0)
    median_segment = float(np.median(segment_lengths)) if segment_lengths else 0.0
    long_threshold = max(float(line_width) * long_segment_factor, median_segment * 4.0)

    outside_segments = 0
    long_outside_segments = 0
    if printable_geometry is not None and not printable_geometry.is_empty:
        padded = printable_geometry.buffer(max(float(line_width), GeometrySettings.COORD_EPSILON) * 0.05)
        for segment in segments:
            line_segment = LineString(segment)
            if not line_segment.covered_by(padded):
                outside_segments += 1
                if line_segment.length > long_threshold:
                    long_outside_segments += 1

    tolerance = max(line_width * 0.01, GeometrySettings.COORD_EPSILON)
    crossing_count, overlap_count = _count_crossings_and_overlaps(segments, tolerance)
    return {
        "point_count": int(sum(len(polyline) for polyline in polylines)),
        "polyline_count": int(len(polylines)),
        "segment_count": int(len(segments)),
        "path_length": _safe_length(line),
        "max_segment_length": max_segment,
        "median_segment_length": median_segment,
        "long_segment_threshold": long_threshold,
        "outside_segment_count": outside_segments,
        "long_outside_segment_count": long_outside_segments,
        "crossing_count": crossing_count,
        "self_intersection_count": crossing_count,
        "self_overlap_count": overlap_count,
        "is_simple_path": bool(line.is_simple) if line is not None else False,
    }


def effective_hole_contour_metrics(
        *,
        path_points: object,
        line_width: float,
        outer_geometry: base.BaseGeometry | None,
        holes: Iterable[base.BaseGeometry],
) -> dict[str, float | int | bool]:
    """Measure the path against the effective domain after unsupported holes."""
    valid_holes = [hole for hole in holes if hole is not None and not hole.is_empty]
    effective = _effective_printable_geometry(outer_geometry, valid_holes)
    if outer_geometry is None or outer_geometry.is_empty or not valid_holes:
        return {
            "effective_printable_area": _safe_area(effective),
            "effective_linear_density_percent": 0.0,
            "effective_covered_density_percent": 0.0,
            "effective_path_fragment_count": 0,
            "effective_path_length_after_hole_cut": 0.0,
            "effective_path_loses_continuity": False,
        }

    line = _line_from_polylines(path_points)
    if line is None or line.is_empty:
        clipped_line = GeometryCollection()
    else:
        clipped_line = line.intersection(effective)

    fragment_count = 0
    if clipped_line.is_empty:
        fragment_count = 0
    elif clipped_line.geom_type in {"LineString", "LinearRing"}:
        fragment_count = 1
    elif clipped_line.geom_type == "MultiLineString":
        fragment_count = len(clipped_line.geoms)
    elif clipped_line.geom_type == "GeometryCollection":
        fragment_count = sum(
            1 for item in clipped_line.geoms
            if item.geom_type in {"LineString", "LinearRing"}
        )

    coverage = coverage_metrics(
        path_points=path_points,
        line_width=line_width,
        printable_geometry=effective,
    )
    return {
        "effective_printable_area": _safe_area(effective),
        "effective_linear_density_percent": coverage["linear_density_percent"],
        "effective_covered_density_percent": coverage["covered_density_percent"],
        "effective_path_fragment_count": fragment_count,
        "effective_path_length_after_hole_cut": _safe_length(clipped_line),
        "effective_path_loses_continuity": fragment_count > 1,
    }


def hole_intrusion_metrics(
        *,
        path_points: object,
        line_width: float,
        holes: Iterable[base.BaseGeometry],
) -> dict[str, float | int | bool]:
    """Measure whether the swept path intrudes into unsupported hole regions."""
    swept = swept_geometry(path_points, line_width)
    total_hole_area = 0.0
    intrusion_area = 0.0
    hole_count = 0
    for hole in holes:
        if hole is None or hole.is_empty:
            continue
        hole_count += 1
        total_hole_area += _safe_area(hole)
        if swept is not None:
            intrusion_area += _safe_area(swept.intersection(hole))

    return {
        "hole_count": hole_count,
        "hole_area": total_hole_area,
        "hole_intrusion_area": intrusion_area,
        "hole_intrusion_percent": (
            100.0 * intrusion_area / total_hole_area
            if total_hole_area > GeometrySettings.AREA_EPSILON
            else 0.0
        ),
        "has_hole_intrusion": intrusion_area > GeometrySettings.AREA_EPSILON,
    }


def peak_spacing_metrics(path_points: object) -> dict[str, float | int]:
    """Estimate motif spacing from local X extrema along the infill centerline."""
    polylines = _as_polylines(path_points)
    extrema_y: list[float] = []
    for points in polylines:
        if len(points) < 3:
            continue
        x_values = points[:, 0]
        for index in range(1, len(points) - 1):
            before = x_values[index] - x_values[index - 1]
            after = x_values[index + 1] - x_values[index]
            if before == 0.0 or after == 0.0:
                continue
            if before * after < 0.0:
                extrema_y.append(float(points[index, 1]))

    extrema_y = sorted(set(round(value, 6) for value in extrema_y))
    spacings = [
        abs(float(end) - float(start))
        for start, end in zip(extrema_y[:-1], extrema_y[1:])
        if abs(float(end) - float(start)) > GeometrySettings.LENGTH_EPSILON
    ]
    return {
        "peak_count": len(extrema_y),
        "peak_spacing_median": float(np.median(spacings)) if spacings else 0.0,
    }


def preview_metrics(
        preview_result: object,
        *,
        holes: Iterable[base.BaseGeometry] = (),
        extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Collect validation metrics from a `_PatternPreviewResult`-like object."""
    line_width = float(getattr(preview_result, "line_width", 0.0))
    infill_points = getattr(preview_result, "infill_path_points", None)
    if infill_points is None or len(_as_polylines(infill_points)) == 0:
        infill_points = getattr(preview_result, "path_points")
    infill_domain = getattr(preview_result, "infill_boundary", None)
    if infill_domain is None or infill_domain.is_empty:
        infill_domain = getattr(preview_result, "final_contour", None)
    effective_infill_domain = _effective_printable_geometry(infill_domain, holes)

    full_path = getattr(preview_result, "path_points", infill_points)
    wall_geometries = list(getattr(preview_result, "wall_geometries", []) or [])
    wall_length = sum(_safe_length(wall) for wall in wall_geometries)
    full_length = _safe_length(_line_from_polylines(full_path))
    infill_length = _safe_length(_line_from_polylines(infill_points))

    metrics: dict[str, object] = {
        "pattern": getattr(preview_result, "infill_pattern", ""),
        "infill_density": float(getattr(preview_result, "infill_density", 0.0)),
        "estimated_density": float(getattr(preview_result, "estimated_density", 0.0)),
        "line_width": line_width,
        "line_distance": float(getattr(preview_result, "line_distance", 0.0)),
        "pattern_distance_factor": float(getattr(preview_result, "pattern_distance_factor", 0.0)),
        "pattern_amplitude": float(getattr(preview_result, "pattern_amplitude", 0.0)),
        "pattern_step": float(getattr(preview_result, "pattern_step", 0.0)),
        "infill_spacing": float(getattr(preview_result, "infill_spacing", 0.0)),
        "fillet_radius": float(getattr(preview_result, "fillet_radius", 0.0)),
        "fillet_radius_over_line_width": (
            float(getattr(preview_result, "fillet_radius", 0.0)) / max(line_width, GeometrySettings.COORD_EPSILON)
        ),
        "num_walls": int(getattr(preview_result, "num_walls", 0)),
        "wall_length": wall_length,
        "full_path_length": full_length,
        "connector_length_estimate": max(0.0, full_length - infill_length - wall_length),
        "density_basis": "infill_centerline_length_x_line_width_over_effective_infill_area",
    }
    metrics.update(coverage_metrics(
        path_points=infill_points,
        line_width=line_width,
        printable_geometry=effective_infill_domain,
    ))
    metrics.update(path_quality_metrics(
        path_points=infill_points,
        printable_geometry=effective_infill_domain,
        line_width=line_width,
    ))
    metrics.update(wall_clearance_metrics(
        path_points=infill_points,
        wall_centerlines=wall_geometries,
        line_width=line_width,
    ))
    metrics.update(eulerian_metrics(
        infill_points,
        tolerance=max(line_width * 0.01, GeometrySettings.COORD_EPSILON),
    ))
    metrics.update(ordered_path_continuity_metrics(
        infill_points,
        tolerance=max(line_width * 0.05, GeometrySettings.COORD_EPSILON),
    ))
    metrics.update(hole_intrusion_metrics(
        path_points=infill_points,
        line_width=line_width,
        holes=holes,
    ))
    metrics.update(effective_hole_contour_metrics(
        path_points=infill_points,
        line_width=line_width,
        outer_geometry=infill_domain,
        holes=holes,
    ))
    metrics.update(peak_spacing_metrics(infill_points))
    if extra:
        metrics.update(dict(extra))
    return metrics

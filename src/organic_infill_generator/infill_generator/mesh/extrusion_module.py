"""
extrusion_module.py
===================
3D extrusion of 2D tool-paths using an explicit ribbon sweep.

The 2D path is rounded first, then each segment is offset left/right by half the
requested width, consecutive offsets are intersected, and the final ribbon is
stitched with quads. This preserves the requested `line_width` much more
faithfully than buffering the whole polyline as a fused area.

Public API:
    generate_mesh(path_2d, height, thickness) -> (vertices, faces)
    generate_solid_contour_mesh(contour_2d, height, fillet_radius=0.0) -> (vertices, faces)
    export_stl(vertices, faces, filename)
"""
from __future__ import annotations


import logging
import math
import os
import struct
from pathlib import Path

import numpy as np
from shapely import set_precision
from shapely.geometry import JOIN_STYLE, LineString, MultiLineString, Polygon
from shapely.ops import linemerge
from shapely.geometry.polygon import orient
from shapely.ops import triangulate

from ...common.geometry import (
    _remove_collinear_points,
    apply_min_corner_radius_to_closed_path,
    apply_min_corner_radius_to_path,
)
from ..settings import FilletSettings, GeometrySettings, MeshSettings


# ======================================================================
# MESH GENERATION
# ======================================================================

MESH_FACE_EPSILON = 1e-12

logger = logging.getLogger(__name__)


def _triangle_area_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the area of a 3D triangle."""
    return 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))


def _filter_mesh_faces(
        vertices: np.ndarray,
        faces: np.ndarray,
        area_epsilon: float = MESH_FACE_EPSILON,
) -> tuple[np.ndarray, int]:
    """
    Remove degenerate triangles from a raw mesh soup.

    A face is discarded when:
    - it repeats any vertex index
    - its geometric 3D area collapses to ~0
    """
    if len(faces) == 0:
        return faces, 0

    keep_mask = np.ones(len(faces), dtype=bool)
    for index, face in enumerate(faces):
        i0, i1, i2 = int(face[0]), int(face[1]), int(face[2])
        if i0 == i1 or i1 == i2 or i0 == i2:
            keep_mask[index] = False
            continue

        area = _triangle_area_3d(vertices[i0], vertices[i1], vertices[i2])
        if area <= area_epsilon:
            keep_mask[index] = False

    removed_count = int((~keep_mask).sum())
    return faces[keep_mask], removed_count


def _iter_polygons(geometry):
    """Yield polygon parts from Polygon / MultiPolygon-like geometries."""
    if geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
    elif geometry.geom_type == "MultiPolygon":
        for polygon in geometry.geoms:
            yield polygon


def _vertex_key(x: float, y: float, z: float) -> tuple[float, float, float]:
    return (round(float(x), 12), round(float(y), 12), round(float(z), 12))


def _filter_path_points(path_2d: np.ndarray) -> np.ndarray:
    if len(path_2d) < 2:
        return np.asarray(path_2d, dtype=float)

    path = [np.asarray(path_2d[0], dtype=float)]
    for index in range(1, len(path_2d)):
        current = np.asarray(path_2d[index], dtype=float)
        delta = current - path[-1]
        if float(np.dot(delta, delta)) > GeometrySettings.LENGTH_EPSILON ** 2:
            path.append(current)
    return np.asarray(path, dtype=float)


def _prepare_mesh_path(
        path_2d: np.ndarray,
        mesh_min_corner_radius: float,
        thickness: float,
        fillet_protected_indices: set[int] | list[int] | tuple[int, ...] | None = None,
) -> np.ndarray:
    path = _filter_path_points(np.asarray(path_2d, dtype=float))
    if len(path) < 2:
        return path

    rounded_path = apply_min_corner_radius_to_path(
        path,
        min_corner_radius=mesh_min_corner_radius,
        min_turn_degrees=FilletSettings.MIN_TURN_DEGREES,
        segments_per_quadrant=FilletSettings.QUAD_SEGS,
        protected_indices=fillet_protected_indices,
    )
    return _filter_path_points(rounded_path)


def _prepare_closed_contour_points(
        contour_2d: np.ndarray,
        fillet_radius: float,
) -> np.ndarray:
    points = np.asarray(contour_2d, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        return np.zeros((0, 2), dtype=float)

    if len(points) >= 2 and np.linalg.norm(points[0] - points[-1]) <= GeometrySettings.COORD_EPSILON:
        points = points[:-1]

    points = _filter_path_points(points)
    if len(points) < 3:
        return np.zeros((0, 2), dtype=float)

    if fillet_radius > GeometrySettings.AREA_EPSILON:
        points = apply_min_corner_radius_to_closed_path(
            points,
            min_corner_radius=fillet_radius,
            min_turn_degrees=FilletSettings.MIN_TURN_DEGREES,
            segments_per_quadrant=FilletSettings.QUAD_SEGS,
        )
        points = _filter_path_points(points)

    return points if len(points) >= 3 else np.zeros((0, 2), dtype=float)


def _normalize_solid_geometry(solid_geometry):
    if solid_geometry is None or solid_geometry.is_empty:
        return None
    if getattr(solid_geometry, "area", 0.0) <= GeometrySettings.AREA_EPSILON:
        return None

    precision_grid = float(MeshSettings.SWEPT_GEOMETRY_PRECISION_GRID)
    if precision_grid > GeometrySettings.AREA_EPSILON:
        try:
            snapped = set_precision(solid_geometry, precision_grid)
            if not snapped.is_empty:
                solid_geometry = snapped
        except Exception:
            pass

    if not solid_geometry.is_valid:
        repaired = solid_geometry.buffer(0)
        if not repaired.is_empty:
            solid_geometry = repaired

    return solid_geometry if not solid_geometry.is_empty else None


def build_solid_contour_geometry(
        contour_2d: np.ndarray,
        fillet_radius: float = 0.0,
):
    """
    Convert a closed 2D contour into a clean polygon suitable for extrusion.
    """
    points = _prepare_closed_contour_points(contour_2d, fillet_radius=fillet_radius)
    if len(points) < 3:
        return None

    solid = Polygon(points)
    return _normalize_solid_geometry(solid)


def build_solid_geometry_from_centerline_contour(
        centerline_contour_2d: np.ndarray,
        *,
        thickness: float,
        fillet_radius: float = 0.0,
):
    """
    Build the full solid footprint implied by a closed wall centerline contour.

    This is useful for top/bottom solid layers in sandwich mode, where we want
    the outer boundary to match the same first-wall centerline + line width
    convention used by the regular infill/wall path.
    """
    if thickness <= GeometrySettings.AREA_EPSILON:
        return None

    points = np.asarray(centerline_contour_2d, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        return None

    if len(points) >= 2 and np.linalg.norm(points[0] - points[-1]) <= GeometrySettings.COORD_EPSILON:
        points = points[:-1]

    points = _filter_path_points(points)
    points = _remove_collinear_points(points)
    if len(points) < 3:
        return None

    if fillet_radius > GeometrySettings.AREA_EPSILON:
        points = apply_min_corner_radius_to_closed_path(
            points,
            min_corner_radius=fillet_radius,
            min_turn_degrees=FilletSettings.MIN_TURN_DEGREES,
            segments_per_quadrant=FilletSettings.QUAD_SEGS,
        )
        points = _filter_path_points(points)
        points = _remove_collinear_points(points)

    if len(points) < 3:
        return None

    centerline_polygon = Polygon(points)
    if centerline_polygon.is_empty or centerline_polygon.area <= GeometrySettings.AREA_EPSILON:
        return None

    join_style = JOIN_STYLE.round if fillet_radius > GeometrySettings.AREA_EPSILON else JOIN_STYLE.mitre
    solid = centerline_polygon.buffer(
        thickness / 2.0,
        join_style=join_style,
        quad_segs=FilletSettings.QUAD_SEGS,
    )
    return _normalize_solid_geometry(solid)


def _build_raw_solid_geometry_mesh(
        solid_geometry,
        height: float,
) -> tuple[np.ndarray, np.ndarray]:
    if height <= GeometrySettings.AREA_EPSILON or solid_geometry is None or solid_geometry.is_empty:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    solid_geometry = _normalize_solid_geometry(solid_geometry)
    if solid_geometry is None:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    mesh_data = _build_extruded_polygon_mesh_with_trimesh(
        solid_geometry,
        height=height,
    )
    if mesh_data is None:
        vertices, faces = _build_extruded_polygon_mesh(
            solid_geometry,
            height=height,
        )
    else:
        vertices, faces = mesh_data

    if len(vertices) == 0 or len(faces) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    return vertices, faces



def _compute_segment_offsets(
        start_point: np.ndarray,
        end_point: np.ndarray,
        half_thickness: float,
) -> dict[str, np.ndarray] | None:
    delta = end_point - start_point
    length = float(np.linalg.norm(delta))
    if length <= GeometrySettings.LENGTH_EPSILON:
        return None

    normal = np.array([-delta[1], delta[0]], dtype=float) / length
    offset = normal * half_thickness
    return {
        "left_start": start_point + offset,
        "left_end": end_point + offset,
        "right_start": start_point - offset,
        "right_end": end_point - offset,
    }


def _intersect_offset_lines(
        line1_start: np.ndarray,
        line1_end: np.ndarray,
        line2_start: np.ndarray,
        line2_end: np.ndarray,
) -> np.ndarray:
    """Intersect two offset centerline segments, falling back to the first end."""
    x1, y1 = line1_start
    x2, y2 = line1_end
    x3, y3 = line2_start
    x4, y4 = line2_end
    denominator = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)

    if abs(denominator) < GeometrySettings.VERTICAL_INTERSECTION_EPSILON:
        return np.asarray(line1_end, dtype=float)

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator
    return np.array([x1 + ua * (x2 - x1), y1 + ua * (y2 - y1)], dtype=float)


def _build_round_cap_points(
        *,
        center: np.ndarray,
        tangent: np.ndarray,
        normal: np.ndarray,
        radius: float,
        arc_steps: int,
) -> list[np.ndarray]:
    outward = tangent / max(float(np.linalg.norm(tangent)), GeometrySettings.LENGTH_EPSILON)
    side = normal / max(float(np.linalg.norm(normal)), GeometrySettings.LENGTH_EPSILON)

    points: list[np.ndarray] = []
    for theta in np.linspace(math.pi / 2.0, -math.pi / 2.0, arc_steps + 1):
        point = center + radius * (math.cos(theta) * outward + math.sin(theta) * side)
        points.append(np.asarray(point, dtype=float))
    return points


def _normalize_offset_curve_geometry(
        geometry,
        path_start: np.ndarray,
) -> list[np.ndarray] | None:
    """Extract an ordered coordinate list from a shapely offset curve result."""
    if geometry.is_empty:
        return None

    if isinstance(geometry, MultiLineString):
        geometry = linemerge(geometry)

    if isinstance(geometry, MultiLineString):
        candidates = [np.asarray(line.coords, dtype=float) for line in geometry.geoms if len(line.coords) >= 2]
        if not candidates:
            return None
        coords = max(candidates, key=lambda item: LineString(item).length)
    elif isinstance(geometry, LineString):
        coords = np.asarray(geometry.coords, dtype=float)
    else:
        return None

    if len(coords) < 2:
        return None

    if np.linalg.norm(coords[0] - path_start) > np.linalg.norm(coords[-1] - path_start):
        coords = coords[::-1]

    filtered: list[np.ndarray] = [coords[0]]
    for point in coords[1:]:
        if np.linalg.norm(point - filtered[-1]) > GeometrySettings.LENGTH_EPSILON:
            filtered.append(point)
    return filtered if len(filtered) >= 2 else None


def _build_ribbon_profiles(
        path_2d: np.ndarray,
        thickness: float,
        mesh_min_corner_radius: float,
        fillet_protected_indices: set[int] | list[int] | tuple[int, ...] | None = None,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]] | None:
    """Build left/right side profiles by offsetting each path segment."""
    path = _prepare_mesh_path(
        path_2d,
        mesh_min_corner_radius=mesh_min_corner_radius,
        thickness=thickness,
        fillet_protected_indices=fillet_protected_indices,
    )
    if len(path) < 2 or thickness <= GeometrySettings.AREA_EPSILON:
        return None

    half_thickness = thickness / 2.0
    segments: list[dict[str, np.ndarray]] = []
    for index in range(len(path) - 1):
        segment = _compute_segment_offsets(path[index], path[index + 1], half_thickness)
        if segment is not None:
            segments.append(segment)

    if not segments:
        return None

    left_profile = [segments[0]["left_start"]]
    right_profile = [segments[0]["right_start"]]

    for previous_segment, next_segment in zip(segments[:-1], segments[1:]):
        left_profile.append(
            _intersect_offset_lines(
                previous_segment["left_start"],
                previous_segment["left_end"],
                next_segment["left_start"],
                next_segment["left_end"],
            )
        )
        right_profile.append(
            _intersect_offset_lines(
                previous_segment["right_start"],
                previous_segment["right_end"],
                next_segment["right_start"],
                next_segment["right_end"],
            )
        )

    left_profile.append(segments[-1]["left_end"])
    right_profile.append(segments[-1]["right_end"])
    return path, left_profile, right_profile


def _build_ribbon_outline_geometry(
        path_2d: np.ndarray,
        thickness: float,
        mesh_min_corner_radius: float,
        fillet_protected_indices: set[int] | list[int] | tuple[int, ...] | None = None,
):
    """Create the 2D footprint of a rounded ribbon before extrusion."""
    path = _prepare_mesh_path(
        path_2d,
        mesh_min_corner_radius=mesh_min_corner_radius,
        thickness=thickness,
        fillet_protected_indices=fillet_protected_indices,
    )
    if len(path) < 2 or thickness <= GeometrySettings.AREA_EPSILON:
        return None

    centerline = LineString(path)
    if centerline.is_empty or centerline.length <= GeometrySettings.LENGTH_EPSILON:
        return None

    half_thickness = thickness / 2.0
    offset_kwargs = {
        "join_style": 1,
        "quad_segs": FilletSettings.QUAD_SEGS,
    }
    left_curve = centerline.offset_curve(half_thickness, **offset_kwargs)
    right_curve = centerline.offset_curve(-half_thickness, **offset_kwargs)

    left_profile = _normalize_offset_curve_geometry(left_curve, path_start=path[0])
    right_profile = _normalize_offset_curve_geometry(right_curve, path_start=path[0])
    if left_profile is None or right_profile is None:
        # Shapely offset_curve can split or drop geometry in tight corners.
        # Segment-wise offsets are simpler but give us a robust fallback.
        profiles = _build_ribbon_profiles(
            path_2d,
            thickness=thickness,
            mesh_min_corner_radius=mesh_min_corner_radius,
            fillet_protected_indices=fillet_protected_indices,
        )
        if profiles is None:
            return None
        path, left_profile, right_profile = profiles

    arc_steps = max(2, FilletSettings.QUAD_SEGS * 2)

    start_cap = _build_round_cap_points(
        center=path[0],
        tangent=path[0] - path[1],
        normal=np.asarray(left_profile[0], dtype=float) - path[0],
        radius=half_thickness,
        arc_steps=arc_steps,
    )
    end_cap = _build_round_cap_points(
        center=path[-1],
        tangent=path[-1] - path[-2],
        normal=np.asarray(left_profile[-1], dtype=float) - path[-1],
        radius=half_thickness,
        arc_steps=arc_steps,
    )

    boundary_points: list[tuple[float, float]] = [tuple(map(float, point)) for point in left_profile]
    boundary_points.extend(tuple(map(float, point)) for point in end_cap[1:])
    boundary_points.extend(tuple(map(float, point)) for point in reversed(right_profile[:-1]))
    boundary_points.extend(tuple(map(float, point)) for point in start_cap[-1:0:-1])

    filtered_boundary: list[tuple[float, float]] = []
    for point in boundary_points:
        if not filtered_boundary:
            filtered_boundary.append(point)
            continue
        delta_x = point[0] - filtered_boundary[-1][0]
        delta_y = point[1] - filtered_boundary[-1][1]
        if math.hypot(delta_x, delta_y) > GeometrySettings.LENGTH_EPSILON:
            filtered_boundary.append(point)

    if len(filtered_boundary) >= 2:
        delta_x = filtered_boundary[0][0] - filtered_boundary[-1][0]
        delta_y = filtered_boundary[0][1] - filtered_boundary[-1][1]
        if math.hypot(delta_x, delta_y) <= GeometrySettings.LENGTH_EPSILON:
            filtered_boundary.pop()

    outline = Polygon(filtered_boundary)
    if outline.is_empty:
        return None

    precision_grid = float(MeshSettings.SWEPT_GEOMETRY_PRECISION_GRID)
    if precision_grid > GeometrySettings.AREA_EPSILON:
        try:
            snapped = set_precision(outline, precision_grid)
            if not snapped.is_empty:
                outline = snapped
        except Exception:
            pass

    if not outline.is_valid:
        repaired = outline.buffer(0)
        if not repaired.is_empty:
            outline = repaired

    return outline if not outline.is_empty else None


def _build_ribbon_mesh(
        path_2d: np.ndarray,
        height: float,
        thickness: float,
        mesh_min_corner_radius: float,
        fillet_protected_indices: set[int] | list[int] | tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fallback ribbon mesh builder that mirrors the old HTML preview geometry."""
    profiles = _build_ribbon_profiles(
        path_2d,
        thickness=thickness,
        mesh_min_corner_radius=mesh_min_corner_radius,
        fillet_protected_indices=fillet_protected_indices,
    )
    if profiles is None:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    path, left_profile, right_profile = profiles
    half_thickness = thickness / 2.0

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    vertex_index: dict[tuple[float, float, float], int] = {}

    def add_vertex(point: np.ndarray, z: float) -> int:
        key = _vertex_key(point[0], point[1], z)
        existing = vertex_index.get(key)
        if existing is not None:
            return existing
        index = len(vertices)
        vertices.append([float(point[0]), float(point[1]), float(z)])
        vertex_index[key] = index
        return index

    def add_quad(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, v4: np.ndarray) -> None:
        i1 = add_vertex(v1[:2], v1[2])
        i2 = add_vertex(v2[:2], v2[2])
        i3 = add_vertex(v3[:2], v3[2])
        i4 = add_vertex(v4[:2], v4[2])
        faces.append([i1, i2, i4])
        faces.append([i2, i3, i4])

    def add_round_cap(
            *,
            center: np.ndarray,
            tangent: np.ndarray,
            normal: np.ndarray,
    ) -> None:
        arc_points = _build_round_cap_points(
            center=center,
            tangent=tangent,
            normal=normal,
            radius=half_thickness,
            arc_steps=max(2, FilletSettings.QUAD_SEGS * 2),
        )

        for index in range(len(arc_points) - 1):
            point_a = arc_points[index]
            point_b = arc_points[index + 1]

            a_bottom = np.array([point_a[0], point_a[1], 0.0], dtype=float)
            a_top = np.array([point_a[0], point_a[1], height], dtype=float)
            b_bottom = np.array([point_b[0], point_b[1], 0.0], dtype=float)
            b_top = np.array([point_b[0], point_b[1], height], dtype=float)

            add_quad(a_bottom, b_bottom, b_top, a_top)

        arc_top = [np.array([point[0], point[1], height], dtype=float) for point in arc_points]
        arc_bottom = [np.array([point[0], point[1], 0.0], dtype=float) for point in arc_points]

        for index in range(1, len(arc_top) - 1):
            i0 = add_vertex(arc_top[0][:2], arc_top[0][2])
            i1 = add_vertex(arc_top[index][:2], arc_top[index][2])
            i2 = add_vertex(arc_top[index + 1][:2], arc_top[index + 1][2])
            faces.append([i0, i1, i2])

            j0 = add_vertex(arc_bottom[0][:2], arc_bottom[0][2])
            j1 = add_vertex(arc_bottom[index][:2], arc_bottom[index][2])
            j2 = add_vertex(arc_bottom[index + 1][:2], arc_bottom[index + 1][2])
            faces.append([j0, j2, j1])

    for index in range(len(left_profile) - 1):
        left_start = left_profile[index]
        left_end = left_profile[index + 1]
        right_start = right_profile[index]
        right_end = right_profile[index + 1]

        left_start_bottom = np.array([left_start[0], left_start[1], 0.0], dtype=float)
        left_start_top = np.array([left_start[0], left_start[1], height], dtype=float)
        left_end_bottom = np.array([left_end[0], left_end[1], 0.0], dtype=float)
        left_end_top = np.array([left_end[0], left_end[1], height], dtype=float)
        right_start_bottom = np.array([right_start[0], right_start[1], 0.0], dtype=float)
        right_start_top = np.array([right_start[0], right_start[1], height], dtype=float)
        right_end_bottom = np.array([right_end[0], right_end[1], 0.0], dtype=float)
        right_end_top = np.array([right_end[0], right_end[1], height], dtype=float)

        add_quad(left_start_top, left_end_top, right_end_top, right_start_top)
        add_quad(left_start_bottom, right_start_bottom, right_end_bottom, left_end_bottom)
        add_quad(left_start_bottom, left_end_bottom, left_end_top, left_start_top)
        add_quad(right_start_bottom, right_start_top, right_end_top, right_end_bottom)

    start_tangent = path[0] - path[1]
    start_normal = left_profile[0] - path[0]
    add_round_cap(
        center=path[0],
        tangent=start_tangent,
        normal=start_normal,
    )

    end_tangent = path[-1] - path[-2]
    end_normal = left_profile[-1] - path[-1]
    add_round_cap(
        center=path[-1],
        tangent=end_tangent,
        normal=end_normal,
    )

    if not vertices or not faces:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def _build_extruded_polygon_mesh(
        swept_geometry,
        height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a watertight indexed mesh by extruding a 2D swept polygon.
    """
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    vertex_index: dict[tuple[float, float, float], int] = {}

    def add_vertex(x: float, y: float, z: float) -> int:
        key = _vertex_key(x, y, z)
        existing = vertex_index.get(key)
        if existing is not None:
            return existing
        index = len(vertices)
        vertices.append([float(x), float(y), float(z)])
        vertex_index[key] = index
        return index

    def add_triangle(p0, p1, p2, z: float, reverse: bool = False) -> None:
        i0 = add_vertex(p0[0], p0[1], z)
        i1 = add_vertex(p1[0], p1[1], z)
        i2 = add_vertex(p2[0], p2[1], z)
        if reverse:
            faces.append([i0, i2, i1])
        else:
            faces.append([i0, i1, i2])

    def add_side_ring(ring_coords: list[tuple[float, float]]) -> None:
        for idx in range(len(ring_coords) - 1):
            p0 = ring_coords[idx]
            p1 = ring_coords[idx + 1]
            b0 = add_vertex(p0[0], p0[1], 0.0)
            b1 = add_vertex(p1[0], p1[1], 0.0)
            t0 = add_vertex(p0[0], p0[1], height)
            t1 = add_vertex(p1[0], p1[1], height)
            faces.append([b0, b1, t1])
            faces.append([b0, t1, t0])

    for polygon in _iter_polygons(swept_geometry):
        polygon = orient(polygon, sign=1.0)

        for triangle in triangulate(polygon):
            if not polygon.covers(triangle):
                continue
            coords = list(triangle.exterior.coords)[:-1]
            if len(coords) != 3:
                continue

            # Top face (+Z)
            add_triangle(coords[0], coords[1], coords[2], height, reverse=False)
            # Bottom face (-Z)
            add_triangle(coords[0], coords[1], coords[2], 0.0, reverse=True)

        add_side_ring(list(polygon.exterior.coords))
        for interior in polygon.interiors:
            add_side_ring(list(interior.coords))

    if not vertices or not faces:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def _build_extruded_polygon_mesh_with_trimesh(
        swept_geometry,
        height: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Use trimesh's polygon extrusion when available.
    """
    try:
        import trimesh
    except ImportError:
        return None

    try:
        mesh = trimesh.creation.extrude_polygon(swept_geometry, height=height)
        return np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=int)
    except Exception:
        return None


def _build_swept_geometry(
        path_2d: np.ndarray,
        thickness: float,
        mesh_min_corner_radius: float,
        fillet_protected_indices: set[int] | list[int] | tuple[int, ...] | None = None,
):
    """
    Convert a rounded 2D centerline path into a swept solid footprint.
    """
    path = _prepare_mesh_path(
        path_2d,
        mesh_min_corner_radius=mesh_min_corner_radius,
        thickness=thickness,
        fillet_protected_indices=fillet_protected_indices,
    )
    if len(path) < 2 or thickness <= GeometrySettings.AREA_EPSILON:
        return None

    swept = LineString(path).buffer(
        thickness / 2.0,
        cap_style=1,
        join_style=1,
        quad_segs=FilletSettings.QUAD_SEGS,
    )
    if swept.is_empty:
        return None

    precision_grid = float(MeshSettings.SWEPT_GEOMETRY_PRECISION_GRID)
    if precision_grid > GeometrySettings.AREA_EPSILON:
        try:
            snapped = set_precision(swept, precision_grid)
            if not snapped.is_empty:
                swept = snapped
        except Exception:
            pass

    if not swept.is_valid:
        repaired = swept.buffer(0)
        if not repaired.is_empty:
            swept = repaired

    return swept if not swept.is_empty else None


def _build_raw_path_mesh(
        path_2d: np.ndarray,
        height: float,
        thickness: float,
        mesh_min_corner_radius: float,
        fillet_protected_indices: set[int] | list[int] | tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a raw 3D ribbon mesh from a 2D centerline path."""
    if len(path_2d) < 2:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    vertices = np.zeros((0, 3))
    faces = np.zeros((0, 3), dtype=int)

    if mesh_min_corner_radius > GeometrySettings.AREA_EPSILON:
        # Preferred route: make a real 2D swept footprint, then extrude it.
        # This gives cleaner top/bottom faces than stitching offsets directly.
        swept_geometry = _build_swept_geometry(
            path_2d,
            thickness=thickness,
            mesh_min_corner_radius=mesh_min_corner_radius,
            fillet_protected_indices=fillet_protected_indices,
        )
        if swept_geometry is not None:
            swept_mesh = _build_extruded_polygon_mesh_with_trimesh(
                swept_geometry,
                height=height,
            )
            if swept_mesh is None:
                swept_mesh = _build_extruded_polygon_mesh(
                    swept_geometry,
                    height=height,
                )
            if swept_mesh is not None:
                vertices, faces = swept_mesh

    if len(vertices) == 0 or len(faces) == 0:
        # Safety net for pathological offset geometries. It preserves output
        # availability even if the richer polygon extrusion path fails.
        vertices, faces = _build_ribbon_mesh(
            path_2d,
            height=height,
            thickness=thickness,
            mesh_min_corner_radius=mesh_min_corner_radius,
            fillet_protected_indices=fillet_protected_indices,
        )

    if len(vertices) == 0 or len(faces) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    return vertices, faces


def _build_raw_solid_contour_mesh(
        contour_2d: np.ndarray,
        height: float,
        fillet_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    solid_geometry = build_solid_contour_geometry(
        contour_2d,
        fillet_radius=fillet_radius,
    )
    return _build_raw_solid_geometry_mesh(
        solid_geometry,
        height=height,
    )


def _translate_mesh_z(
        vertices: np.ndarray,
        z_offset: float,
) -> np.ndarray:
    if len(vertices) == 0 or abs(z_offset) <= GeometrySettings.AREA_EPSILON:
        return np.asarray(vertices, dtype=float)

    translated = np.asarray(vertices, dtype=float).copy()
    translated[:, 2] += float(z_offset)
    return translated


def _concatenate_mesh_parts(
        mesh_parts: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    vertices_parts: list[np.ndarray] = []
    faces_parts: list[np.ndarray] = []
    vertex_offset = 0

    for vertices, faces in mesh_parts:
        if len(vertices) == 0 or len(faces) == 0:
            continue
        vertices_arr = np.asarray(vertices, dtype=float)
        faces_arr = np.asarray(faces, dtype=int)
        vertices_parts.append(vertices_arr)
        faces_parts.append(faces_arr + vertex_offset)
        vertex_offset += len(vertices_arr)

    if not vertices_parts or not faces_parts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    return np.vstack(vertices_parts), np.vstack(faces_parts)


def _finalize_mesh(
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        log_prefix: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop degenerate faces and orient normals before export."""
    if len(vertices) == 0 or len(faces) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    faces, removed_faces = _filter_mesh_faces(vertices, faces)
    if removed_faces > 0:
        logger.info("[%s] Removed %d degenerate faces before export.", log_prefix, removed_faces)

    try:
        import trimesh
    except ImportError:
        trimesh = None

    if trimesh is not None and len(vertices) > 0 and len(faces) > 0:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        trimesh.repair.fix_normals(mesh)
        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=int)

    return vertices, faces


def generate_mesh(
        path_2d: np.ndarray,
        height: float = 1.0,
        thickness: float = 0.4,
        mesh_min_corner_radius: float = 0.0,
        fillet_protected_indices: set[int] | list[int] | tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a 3D mesh by extruding a 2D path.

    Parameters
    ----------
    path_2d : (N, 2) array
        Ordered 2D coordinates of the tool-path.
    height : float
        Extrusion height along Z.
    thickness : float
        Width of the extruded ribbon / pipe.

    Returns
    -------
    vertices : (M, 3) float array
    faces    : (F, 3) int array   - triangle indices into *vertices*
    """
    vertices, faces = _build_raw_path_mesh(
        path_2d,
        height=height,
        thickness=thickness,
        mesh_min_corner_radius=mesh_min_corner_radius,
        fillet_protected_indices=fillet_protected_indices,
    )
    if len(vertices) == 0 or len(faces) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    return _finalize_mesh(
        vertices,
        faces,
        log_prefix="generate_mesh",
    )


def generate_solid_contour_mesh(
        contour_2d: np.ndarray,
        height: float = 1.0,
        fillet_radius: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a solid 3D mesh by extruding a closed 2D contour.

    Parameters
    ----------
    contour_2d : (N, 2) array
        Ordered 2D coordinates describing a closed contour.
    height : float
        Extrusion height along Z.
    fillet_radius : float
        Minimum corner radius applied to the contour before extrusion.
    """
    vertices, faces = _build_raw_solid_contour_mesh(
        contour_2d,
        height=height,
        fillet_radius=fillet_radius,
    )
    if len(vertices) == 0 or len(faces) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    return _finalize_mesh(
        vertices,
        faces,
        log_prefix="generate_solid_contour_mesh",
    )


def generate_sandwich_mesh(
        *,
        contour_2d: np.ndarray,
        infill_path_2d: np.ndarray,
        total_height: float,
        thickness: float,
        fillet_radius: float = 0.0,
        bottom_layers: int = 0,
        top_layers: int = 0,
        layer_height: float = 0.2,
        fillet_protected_indices: set[int] | list[int] | tuple[int, ...] | None = None,
        solid_footprint_geometry=None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a single mesh that stacks bottom solid layers, middle infill and top
    solid layers into one model.
    """
    if total_height <= GeometrySettings.AREA_EPSILON:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    if thickness <= GeometrySettings.AREA_EPSILON:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    if layer_height <= GeometrySettings.AREA_EPSILON:
        raise ValueError("'layer_height' must be > 0.")
    if bottom_layers < 0:
        raise ValueError("'bottom_layers' must be >= 0.")
    if top_layers < 0:
        raise ValueError("'top_layers' must be >= 0.")

    bottom_height = float(bottom_layers * layer_height)
    top_height = float(top_layers * layer_height)
    infill_height = float(total_height - bottom_height - top_height)
    if infill_height < -GeometrySettings.AREA_EPSILON:
        raise ValueError("Bottom/top solid layers exceed the requested total height.")
    infill_height = max(0.0, infill_height)

    if bottom_height <= GeometrySettings.AREA_EPSILON and top_height <= GeometrySettings.AREA_EPSILON:
        return generate_mesh(
            infill_path_2d,
            height=total_height,
            thickness=thickness,
            mesh_min_corner_radius=fillet_radius,
            fillet_protected_indices=fillet_protected_indices,
        )

    if solid_footprint_geometry is None:
        solid_footprint_geometry = build_solid_contour_geometry(
            contour_2d,
            fillet_radius=fillet_radius,
        )

    if infill_height <= GeometrySettings.AREA_EPSILON:
        vertices, faces = _build_raw_solid_geometry_mesh(
            solid_footprint_geometry,
            height=total_height,
        )
        if len(vertices) == 0 or len(faces) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
        return _finalize_mesh(
            vertices,
            faces,
            log_prefix="generate_sandwich_mesh",
            )

    mesh_parts: list[tuple[np.ndarray, np.ndarray]] = []

    if bottom_height > GeometrySettings.AREA_EPSILON:
        bottom_vertices, bottom_faces = _build_raw_solid_geometry_mesh(
            solid_footprint_geometry,
            height=bottom_height,
        )
        if len(bottom_vertices) > 0 and len(bottom_faces) > 0:
            mesh_parts.append((bottom_vertices, bottom_faces))

    infill_vertices, infill_faces = _build_raw_path_mesh(
        infill_path_2d,
        height=infill_height,
        thickness=thickness,
        mesh_min_corner_radius=fillet_radius,
        fillet_protected_indices=fillet_protected_indices,
    )
    if len(infill_vertices) > 0 and len(infill_faces) > 0:
        mesh_parts.append((
            _translate_mesh_z(infill_vertices, bottom_height),
            infill_faces,
        ))

    if top_height > GeometrySettings.AREA_EPSILON:
        top_vertices, top_faces = _build_raw_solid_geometry_mesh(
            solid_footprint_geometry,
            height=top_height,
        )
        if len(top_vertices) > 0 and len(top_faces) > 0:
            mesh_parts.append((
                _translate_mesh_z(top_vertices, bottom_height + infill_height),
                top_faces,
            ))

    vertices, faces = _concatenate_mesh_parts(mesh_parts)
    if len(vertices) == 0 or len(faces) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    return _finalize_mesh(
        vertices,
        faces,
        log_prefix="generate_sandwich_mesh",
    )


# ======================================================================
# STL EXPORT
# ======================================================================

def export_stl(
        vertices: np.ndarray,
        faces: np.ndarray,
        filename: str = "output.stl",
) -> None:
    """Write a binary STL file from vertices and face indices."""
    _write_binary_stl(vertices, faces, filename)
    resolved_path = os.path.abspath(filename)
    logger.info("Saved mesh with %d triangles: %s", len(faces), resolved_path)


def _write_binary_stl(
        vertices: np.ndarray,
        faces: np.ndarray,
        filename: str | Path,
) -> None:
    """Write a binary STL without applying any cleanup or reporting."""
    num_tri = len(faces)
    header = b"\x00" * 80
    with open(filename, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", num_tri))
        for face in faces:
            # Normal placeholder
            f.write(struct.pack("<fff", 0.0, 0.0, 0.0))
            for v_idx in face:
                v = vertices[v_idx]
                f.write(struct.pack("<fff", float(v[0]), float(v[1]), float(v[2])))
            f.write(struct.pack("<H", 0))

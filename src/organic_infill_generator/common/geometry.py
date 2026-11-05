"""
geometry.py
===========
Shared geometry helper functions used across the package.

Responsibilities:
    - Create Shapely Polygon from point arrays
    - Reorder closed LineString (ring) to start at a specific point
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from shapely.geometry import Polygon, LineString, MultiPoint, Point, JOIN_STYLE, base
from shapely.ops import substring, linemerge

from ..infill_generator.settings import (
    FilletSettings,
    GeometrySettings,
    RingSettings,
)

# ----------------------------------------------------------------------
# Polygon creation
# ----------------------------------------------------------------------

def polygon_from_points(points: np.ndarray | Iterable[tuple[float, float]]) -> Polygon:
    """Create a Shapely Polygon from an array or iterable of (x, y) points."""
    pts = np.asarray(points, dtype=float)
    return Polygon(pts)


def apply_min_inner_corner_radius(
        geometry: base.BaseGeometry,
        min_inner_corner_radius: float,
) -> base.BaseGeometry:
    """
    Round sharp inner corners by applying a morphological closing operation.

    This reflects a practical FDM rule: internal corners sharper than the
    nozzle diameter are hard to print cleanly, so we smooth them before wall
    and infill generation. The operation is:

    1. outward buffer by the target radius
    2. inward buffer by the same radius

    That preferentially removes narrow concave corners and replaces them with
    rounded fillets. If the operation collapses the geometry, the original
    geometry is returned unchanged.
    """
    if geometry.is_empty or min_inner_corner_radius <= GeometrySettings.AREA_EPSILON:
        return geometry

    rounded = geometry.buffer(
        min_inner_corner_radius,
        join_style=JOIN_STYLE.round,
        quad_segs=FilletSettings.QUAD_SEGS,
    ).buffer(
        -min_inner_corner_radius,
        join_style=JOIN_STYLE.round,
        quad_segs=FilletSettings.QUAD_SEGS,
    )

    if rounded.is_empty:
        return geometry

    if rounded.geom_type == "MultiPolygon":
        return max(rounded.geoms, key=lambda item: item.area)

    return rounded




def _point_bucket(point: np.ndarray, tolerance: float) -> tuple[int, int]:
    scale = 1.0 / max(tolerance, GeometrySettings.COORD_EPSILON)
    return int(round(float(point[0]) * scale)), int(round(float(point[1]) * scale))


# The plain formula sqrt(x*x + y*y) and np.linalg.norm (sqrt of a BLAS dot) can
# round differently in the last bit, so a norm used IN arithmetic stays
# np.linalg.norm.  A norm only COMPARED with a limit is decided by the plain
# formula when it is clearly off the limit (by far more than any rounding), and
# by np.linalg.norm otherwise: the same decision, without a numpy call per point.
_NORM_BAND = 1e-9


def _norm_cmp(d, limit: float) -> int:
    """-1, 0 or +1 as np.linalg.norm(d) is below, equal to or above ``limit``."""
    x, y = float(d[0]), float(d[1])
    s = math.sqrt(x * x + y * y)
    margin = abs(limit) * _NORM_BAND
    if s > limit + margin:
        return 1
    if s < limit - margin:
        return -1
    exact = float(np.linalg.norm(d))
    return (exact > limit) - (exact < limit)


def _remove_collinear_points(
        pts: np.ndarray,
        angle_tolerance: float = 1e-3,
        cos_tolerance: float = 0.99,
) -> np.ndarray:
    """Remove nearly-collinear interior points from a polyline."""
    if pts.shape[0] < 3:
        return pts
    kept = [0]
    for i in range(1, len(pts) - 1):
        v1 = pts[i] - pts[kept[-1]]
        v2 = pts[i + 1] - pts[i]
        l1 = np.linalg.norm(v1)
        l2 = np.linalg.norm(v2)
        if l1 > 0 and l2 > 0:
            sin_a = (v1[0] * v2[1] - v1[1] * v2[0]) / (l1 * l2)
            cos_a = (v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)
            if abs(sin_a) < angle_tolerance and cos_a > cos_tolerance:
                continue
        kept.append(i)
    if len(pts) - 1 not in kept:
        kept.append(len(pts) - 1)
    return pts[kept]


def _line_intersection(
        s1: tuple[np.ndarray, np.ndarray],
        s2: tuple[np.ndarray, np.ndarray],
) -> np.ndarray | None:
    """Intersection of two infinite lines, each defined by two points."""
    (p1, p2), (p3, p4) = s1, s2
    x1, y1 = p1; x2, y2 = p2
    x3, y3 = p3; x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < GeometrySettings.LENGTH_EPSILON:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return np.array([px, py])


def _arc_fits_within_local_shape(
        arc_points: np.ndarray,
        source_points: list[np.ndarray],
) -> bool:
    """
    Check whether a candidate fillet arc stays within the local source shape
    envelope implied by the original polyline points.
    """
    if len(source_points) < 3:
        return True

    hull = MultiPoint([tuple(map(float, point)) for point in source_points]).convex_hull
    if hull.is_empty or hull.geom_type in {"Point", "LineString"}:
        return True

    padded_hull = hull.buffer(GeometrySettings.COORD_EPSILON * 10.0)
    return all(
        padded_hull.covers(Point(float(point[0]), float(point[1])))
        for point in np.asarray(arc_points, dtype=float)
    )


def _arc_points_unlimited(
        p0: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        radius: float,
        num_points: int = 10,
) -> np.ndarray:
    """Circular arc at vertex p1 with the given radius (no clamping)."""
    v1 = p0 - p1; v2 = p2 - p1
    l1 = np.linalg.norm(v1); l2 = np.linalg.norm(v2)
    if l1 < GeometrySettings.LENGTH_EPSILON or l2 < GeometrySettings.LENGTH_EPSILON:
        return np.atleast_2d(p1)
    u1 = v1 / l1; u2 = v2 / l2
    dot = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
    theta = math.acos(dot)
    if theta < 1e-6 or math.pi - theta < 1e-6:
        return np.atleast_2d(p1)
    alpha = theta / 2.0
    T = radius / math.tan(alpha)
    b = u1 + u2
    lb = np.linalg.norm(b)
    if lb < GeometrySettings.LENGTH_EPSILON:
        return np.atleast_2d(p1)
    b = b / lb
    C = p1 + (radius / math.sin(alpha)) * b
    A = p1 + T * u1; B = p1 + T * u2
    angleA = math.atan2(A[1] - C[1], A[0] - C[0])
    angleB = math.atan2(B[1] - C[1], B[0] - C[0])
    diff = (angleB - angleA + math.pi) % (2 * math.pi) - math.pi
    angles = [angleA + (i / num_points) * diff for i in range(num_points + 1)]
    # the same scalar products and sums as building each point on its own, done
    # in one array: elementwise float64 operations round identically
    unit = np.array([(math.cos(a), math.sin(a)) for a in angles])
    return C + radius * unit


def _arc_points_clamped(
        p0: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        radius: float,
        num_points: int = 10,
) -> np.ndarray:
    """Circular arc at vertex p1, clamped so the tangent length does not
    exceed half the adjacent segment lengths."""
    v1 = p0 - p1; v2 = p2 - p1
    l1 = np.linalg.norm(v1); l2 = np.linalg.norm(v2)
    if l1 < GeometrySettings.LENGTH_EPSILON or l2 < GeometrySettings.LENGTH_EPSILON:
        return np.atleast_2d(p1)
    u1 = v1 / l1; u2 = v2 / l2
    dot = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
    theta = math.acos(dot)
    if theta < 1e-6 or math.pi - theta < 1e-6:
        return np.atleast_2d(p1)
    alpha = theta / 2.0
    T = radius / math.tan(alpha)
    if T > l1 / 2.0 or T > l2 / 2.0:
        T = min(l1 / 2.0, l2 / 2.0)
        radius = T * math.tan(alpha)
    b = u1 + u2
    lb = np.linalg.norm(b)
    if lb < GeometrySettings.LENGTH_EPSILON:
        return np.atleast_2d(p1)
    b = b / lb
    C = p1 + (radius / math.sin(alpha)) * b
    A = p1 + T * u1; B = p1 + T * u2
    angleA = math.atan2(A[1] - C[1], A[0] - C[0])
    angleB = math.atan2(B[1] - C[1], B[0] - C[0])
    diff = (angleB - angleA + math.pi) % (2 * math.pi) - math.pi
    angles = [angleA + (i / num_points) * diff for i in range(num_points + 1)]
    # the same scalar products and sums as building each point on its own, done
    # in one array: elementwise float64 operations round identically
    unit = np.array([(math.cos(a), math.sin(a)) for a in angles])
    return C + radius * unit


def _apply_min_corner_radius_to_path_segment(
        points: np.ndarray,
        min_corner_radius: float,
        min_turn_degrees: float,
        segments_per_quadrant: int,
) -> np.ndarray:
    """
    Smart fillet: replace sharp polyline corners with circular arcs while
    distinguishing noisy short segments (discretisation artifacts) from
    structural features (walls, deliberate steps).

    Strategy (derived from fillet_gen.py):
    1. Remove collinear points.
    2. Separate long segments (>= 2 * radius) from short ones.
    3. For each junction between consecutive long segments:
       - If the skipped short segments are "noise" (virtual intersection is
         close), replace them with a single enlarged arc that covers their
         bounding box.
       - If they are structural (virtual intersection is far), preserve the
         original short segments with individual arcs.
    4. Apply clamped arcs to all virtual vertices.
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3 or min_corner_radius <= GeometrySettings.AREA_EPSILON:
        return pts

    reduced = _remove_collinear_points(pts)
    if reduced.shape[0] < 3:
        return reduced

    R = min_corner_radius
    arc_n = max(segments_per_quadrant, 2)
    threshold = 2.0 * R

    # -- Identify long segments and groups of skipped short segments --
    long_segments: list[tuple[np.ndarray, np.ndarray]] = []
    skipped_groups: list[list[np.ndarray]] = []
    current_skipped: list[np.ndarray] = []

    for i in range(len(reduced) - 1):
        p1 = reduced[i]
        p2 = reduced[i + 1]
        if _norm_cmp(p2 - p1, threshold) >= 0:
            long_segments.append((p1, p2))
            skipped_groups.append(current_skipped)
            current_skipped = []
        else:
            if len(current_skipped) == 0:
                current_skipped.append(p1)
            current_skipped.append(p2)
    trailing_skipped = current_skipped

    # If no long segments found, fall back to simple per-vertex arcs.
    if len(long_segments) < 2:
        virtual_pts: list[dict] = [{"pt": reduced[0], "r_min": 0.0}]
        for i in range(1, len(reduced) - 1):
            virtual_pts.append({"pt": reduced[i], "r_min": 0.0})
        virtual_pts.append({"pt": reduced[-1], "r_min": 0.0})
    else:
        # -- Build virtual vertex list --
        virtual_pts = []
        leading_skipped = skipped_groups[0] if skipped_groups else []
        if leading_skipped:
            for pt in leading_skipped:
                candidate = np.asarray(pt, dtype=float)
                if (
                    not virtual_pts
                    or _norm_cmp(
                        candidate - virtual_pts[-1]["pt"],
                        GeometrySettings.COORD_EPSILON,
                    )
                    > 0
                ):
                    virtual_pts.append({"pt": candidate, "r_min": 0.0})
        else:
            virtual_pts.append({"pt": long_segments[0][0], "r_min": 0.0})
        noise_distance_factor = 4.0

        for i in range(1, len(long_segments)):
            s_prev = long_segments[i - 1]
            s_curr = long_segments[i]
            skipped = skipped_groups[i]

            V = _line_intersection(s_prev, s_curr)

            # Noise vs structural: if virtual intersection is close to both
            # segment endpoints, the short segments are discretisation noise.
            is_safe_to_skip = False
            if V is not None and len(skipped) > 2:
                if (
                    _norm_cmp(V - s_prev[1], noise_distance_factor * R) < 0
                    and _norm_cmp(V - s_curr[0], noise_distance_factor * R) < 0
                ):
                    is_safe_to_skip = True

            if len(skipped) == 0:
                # Direct junction -- use virtual intersection or endpoint.
                if V is not None:
                    virtual_pts.append({"pt": V, "r_min": 0.0})
                else:
                    virtual_pts.append({"pt": s_curr[0], "r_min": 0.0})

            elif is_safe_to_skip:
                # Noisy corner: compute enlarged arc radius from bounding box.
                skipped_arr = np.array(skipped)
                bb_min = skipped_arr.min(axis=0)
                bb_max = skipped_arr.max(axis=0)

                arc_r1 = _arc_points_unlimited(s_prev[0], V, s_curr[1], 1.0, num_points=10)
                r_req = 0.0
                for pt1 in arc_r1:
                    dx = pt1[0] - V[0]; dy = pt1[1] - V[1]
                    if V[0] > bb_max[0] + 1e-5 and dx < -1e-5:
                        r_req = max(r_req, (bb_max[0] - V[0]) / dx)
                    if V[0] < bb_min[0] - 1e-5 and dx > 1e-5:
                        r_req = max(r_req, (bb_min[0] - V[0]) / dx)
                    if V[1] > bb_max[1] + 1e-5 and dy < -1e-5:
                        r_req = max(r_req, (bb_max[1] - V[1]) / dy)
                    if V[1] < bb_min[1] - 1e-5 and dy > 1e-5:
                        r_req = max(r_req, (bb_min[1] - V[1]) / dy)

                source_shape_points = [
                    np.asarray(s_prev[1], dtype=float),
                    *[np.asarray(point, dtype=float) for point in skipped],
                    np.asarray(s_curr[0], dtype=float),
                ]
                chosen_radius = max(R, r_req)
                candidate_arc = _arc_points_clamped(
                    s_prev[0],
                    V,
                    s_curr[1],
                    chosen_radius,
                    num_points=arc_n,
                )
                if not _arc_fits_within_local_shape(candidate_arc, source_shape_points):
                    candidate_arc = _arc_points_clamped(
                        s_prev[0],
                        V,
                        s_curr[1],
                        R,
                        num_points=arc_n,
                    )
                    chosen_radius = R

                if _arc_fits_within_local_shape(candidate_arc, source_shape_points):
                    virtual_pts.append({"pt": V, "r_min": chosen_radius})
                else:
                    virtual_pts.append({"pt": s_prev[1], "r_min": 0.0})
                    for pt in skipped[1:]:
                        virtual_pts.append({"pt": pt, "r_min": 0.0})

            else:
                # Structural feature: preserve original short segments.
                virtual_pts.append({"pt": s_prev[1], "r_min": 0.0})
                for pt in skipped[1:]:
                    virtual_pts.append({"pt": pt, "r_min": 0.0})

        final_points = [long_segments[-1][1]]
        if trailing_skipped:
            final_points.extend(trailing_skipped[1:])
        for pt in final_points:
            candidate = np.asarray(pt, dtype=float)
            if (
                not virtual_pts
                or _norm_cmp(
                    candidate - virtual_pts[-1]["pt"], GeometrySettings.COORD_EPSILON
                )
                > 0
            ):
                virtual_pts.append({"pt": candidate, "r_min": 0.0})

    # -- Final path: apply arcs to each interior virtual vertex --
    result: list[np.ndarray] = [virtual_pts[0]["pt"]]
    for i in range(1, len(virtual_pts) - 1):
        p0 = virtual_pts[i - 1]["pt"]
        p1 = virtual_pts[i]["pt"]
        p2 = virtual_pts[i + 1]["pt"]
        r_eff = max(R, virtual_pts[i]["r_min"])
        arc = _arc_points_clamped(p0, p1, p2, r_eff, num_points=arc_n)
        for pt in arc:
            if (
                len(result) == 0
                or _norm_cmp(pt - result[-1], GeometrySettings.COORD_EPSILON) > 0
            ):
                result.append(pt)

    last = virtual_pts[-1]["pt"]
    if (
        len(result) == 0
        or _norm_cmp(last - result[-1], GeometrySettings.COORD_EPSILON) > 0
    ):
        result.append(last)

    return np.asarray(result, dtype=float)


def apply_min_corner_radius_to_path(
        points: np.ndarray,
        min_corner_radius: float,
        min_turn_degrees: float = 8.0,
        segments_per_quadrant: int = 8,
        protected_indices: set[int] | list[int] | tuple[int, ...] | None = None,
) -> np.ndarray:
    """
    Replace sharp polyline corners with circular fillets while preserving
    revisited path vertices as protected seam points.

    The full toolpath can legitimately visit the same XY coordinate more than
    once when it closes a wall loop and then continues elsewhere. Applying a
    regular fillet straight through those seam points produces doubled contours
    and breaks the intended single-stroke topology. To avoid that, the path is
    split into independent chunks at every repeated coordinate and each chunk is
    filleted separately.
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3 or min_corner_radius <= GeometrySettings.AREA_EPSILON:
        return pts

    explicit_protected = {
        int(index)
        for index in (protected_indices or [])
        if 0 <= int(index) < len(pts)
    }

    buckets: dict[tuple[int, int], list[int]] = {}
    for index, point in enumerate(pts):
        bucket = _point_bucket(point, GeometrySettings.COORD_EPSILON)
        buckets.setdefault(bucket, []).append(index)

    protected_points = {0, len(pts) - 1, *explicit_protected}
    for indices in buckets.values():
        if len(indices) > 1:
            protected_points.update(indices)

    boundaries = sorted(protected_points)
    if boundaries == [0, len(pts) - 1]:
        return _apply_min_corner_radius_to_path_segment(
            pts,
            min_corner_radius=min_corner_radius,
            min_turn_degrees=min_turn_degrees,
            segments_per_quadrant=segments_per_quadrant,
        )

    rounded_chunks: list[np.ndarray] = []
    for start_index, end_index in zip(boundaries[:-1], boundaries[1:]):
        chunk = pts[start_index:end_index + 1]
        if chunk.shape[0] < 2:
            continue
        rounded_chunk = _apply_min_corner_radius_to_path_segment(
            chunk,
            min_corner_radius=min_corner_radius,
            min_turn_degrees=min_turn_degrees,
            segments_per_quadrant=segments_per_quadrant,
        )
        rounded_chunks.append(rounded_chunk)

    if not rounded_chunks:
        return pts

    merged: list[np.ndarray] = [rounded_chunks[0][0]]
    for chunk_index, rounded_chunk in enumerate(rounded_chunks):
        chunk_points = rounded_chunk if chunk_index == 0 else rounded_chunk[1:]
        for point in chunk_points:
            if np.linalg.norm(point - merged[-1]) > GeometrySettings.COORD_EPSILON:
                merged.append(point)

    return np.asarray(merged, dtype=float)


def apply_min_corner_radius_to_closed_path(
        points: np.ndarray,
        min_corner_radius: float,
        min_turn_degrees: float = 8.0,
        segments_per_quadrant: int = 8,
) -> np.ndarray:
    """
    Apply the existing path fillet logic to a closed contour.

    The seam is moved to the midpoint of the longest edge so it sits on a
    straight segment instead of a real polygon corner. That lets the open-path
    fillet helper round every actual contour vertex.
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] >= 2 and np.linalg.norm(pts[0] - pts[-1]) <= GeometrySettings.COORD_EPSILON:
        pts = pts[:-1]

    if pts.shape[0] < 3 or min_corner_radius <= GeometrySettings.AREA_EPSILON:
        return pts

    filtered: list[np.ndarray] = [pts[0]]
    for point in pts[1:]:
        if np.linalg.norm(point - filtered[-1]) > GeometrySettings.COORD_EPSILON:
            filtered.append(point)
    pts = np.asarray(filtered, dtype=float)

    if pts.shape[0] < 3:
        return pts

    next_points = np.roll(pts, -1, axis=0)
    edge_lengths = np.linalg.norm(next_points - pts, axis=1)
    seam_index = int(np.argmax(edge_lengths))
    if edge_lengths[seam_index] <= GeometrySettings.LENGTH_EPSILON:
        return pts

    seam_point = 0.5 * (pts[seam_index] + pts[(seam_index + 1) % len(pts)])
    reopened: list[np.ndarray] = [seam_point]
    start_index = (seam_index + 1) % len(pts)
    for offset in range(len(pts)):
        reopened.append(pts[(start_index + offset) % len(pts)])
    reopened.append(seam_point)

    rounded = apply_min_corner_radius_to_path(
        np.asarray(reopened, dtype=float),
        min_corner_radius=min_corner_radius,
        min_turn_degrees=min_turn_degrees,
        segments_per_quadrant=segments_per_quadrant,
    )

    if rounded.shape[0] >= 2 and np.linalg.norm(rounded[0] - rounded[-1]) <= GeometrySettings.COORD_EPSILON:
        rounded = rounded[:-1]

    if rounded.shape[0] == 0:
        return pts

    merged: list[np.ndarray] = [rounded[0]]
    for point in rounded[1:]:
        if np.linalg.norm(point - merged[-1]) > GeometrySettings.COORD_EPSILON:
            merged.append(point)

    return np.asarray(merged, dtype=float)


# ----------------------------------------------------------------------
# Ring Reordering
# ----------------------------------------------------------------------

def reorder_ring(ring: LineString, start_point: Point) -> LineString:
    """
    Reorder a closed LineString to start at the point on the ring
    closest to *start_point*.
    """
    if not ring.is_ring:
        pass  # Accept non-strictly-closed linestrings too

    # Find distance along line
    d = ring.project(start_point)
    tol = ring.length * RingSettings.REORDER_TOLERANCE_RATIO
    if d <= tol or d >= ring.length - tol:
        return ring

    # Split ring at d
    part1 = substring(ring, d, ring.length)
    part2 = substring(ring, 0, d)

    c1 = list(part1.coords)
    c2 = list(part2.coords)

    # Remove duplicate join point at the seam
    if len(c1) >= 1 and len(c2) >= 1 and np.allclose(c1[-1], c2[0], atol=GeometrySettings.COORD_EPSILON):
        c2 = c2[1:]

    new_coords = c1 + c2

    # Close the ring: first == last
    if len(new_coords) >= 2 and not np.allclose(new_coords[0], new_coords[-1], atol=GeometrySettings.COORD_EPSILON):
        new_coords.append(new_coords[0])

    return LineString(new_coords)

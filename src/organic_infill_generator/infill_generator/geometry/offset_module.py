"""
offset_module.py
================
Wall-centerline generation and geometric offsets using the new API
semantics.
"""
from __future__ import annotations

from logging import getLogger
from typing import Literal

from shapely.geometry import GeometryCollection, LineString, Point, base
from shapely.geometry import JOIN_STYLE

from ..settings import ApiDefaults, GeometrySettings, PlotSettings, WallSettings

logger = getLogger(__name__)


JOIN_STYLE_MAP = {
    "round": JOIN_STYLE.round,
    "mitre": JOIN_STYLE.mitre,
    "bevel": JOIN_STYLE.bevel,
}


def offset_geometry(
        geometry: base.BaseGeometry,
        offset_distance: float,
        wall_join_style: Literal["round", "mitre", "bevel"] = "round",
) -> base.BaseGeometry:
    """
    Buffer ``geometry`` by ``offset_distance`` using the given join style.
    """
    if geometry.is_empty:
        return geometry

    join_style = JOIN_STYLE_MAP.get(wall_join_style)
    if join_style is None:
        raise ValueError(
            f"wall_join_style must be 'round', 'mitre', or 'bevel', got '{wall_join_style}'."
        )

    return geometry.buffer(offset_distance, join_style=join_style)


def generate_wall_centerlines(
        final_contour: base.BaseGeometry,
        num_walls: int = ApiDefaults.NUM_WALLS,
        line_width: float = ApiDefaults.LINE_WIDTH,
        wall_spacing: float = WallSettings.SPACING,
        wall_join_style: Literal["round", "mitre", "bevel"] = "round",
) -> tuple[list[base.BaseGeometry], base.BaseGeometry]:
    """
    Generate printable wall centerlines plus the infill clip boundary.

    ``final_contour`` represents the final outer dimension of the part.
    The first printed wall is moved inward by ``line_width / 2`` so the final
    deposited material preserves the target dimensions.
    """
    if num_walls < 0:
        raise ValueError("num_walls must be >= 0.")
    if line_width <= 0:
        raise ValueError("line_width must be > 0.")
    if wall_spacing <= -line_width:
        raise ValueError("wall_spacing must be greater than -line_width.")

    wall_centerlines: list[base.BaseGeometry] = []
    # Negative wall_spacing intentionally produces wall overlap by reducing the
    # centerline step below the physical line width.
    centerline_step = line_width + wall_spacing

    for wall_index in range(num_walls):
        distance = -(line_width / 2.0 + wall_index * centerline_step)
        wall_geometry = offset_geometry(
            final_contour,
            distance,
            wall_join_style=wall_join_style,
        )
        if wall_geometry.is_empty or getattr(wall_geometry, "area", 0.0) < GeometrySettings.AREA_EPSILON:
            logger.warning(
                "wall %d collapsed while offsetting (distance=%.3f).",
                wall_index + 1,
                distance,
            )
            continue
        wall_centerlines.append(wall_geometry)

    # The infill clipping contour follows the same centerline offset logic as
    # printed walls: it is the next "virtual wall" after the innermost wall.
    # Example:
    #   num_walls = 2  -> use the centerline position of wall #3 for clipping.
    infill_distance = -(line_width / 2.0 + num_walls * centerline_step)
    infill_boundary = offset_geometry(
        final_contour,
        infill_distance,
        wall_join_style=wall_join_style,
    )

    if infill_boundary.is_empty:
        infill_boundary = GeometryCollection()

    return wall_centerlines, infill_boundary


def create_connection_line(
        original_contour: base.BaseGeometry,
        offset_contour_geometry: base.BaseGeometry,
        angle_degrees: float = 0.0,
) -> tuple[LineString, Point]:
    """
    Create a short straight line connecting an offset contour to the original.
    """
    if offset_contour_geometry.is_empty or original_contour.is_empty:
        if not original_contour.is_empty:
            centroid = original_contour.centroid
            return LineString(), Point(centroid.x, centroid.y)
        return LineString(), Point(0, 0)

    cx, cy = offset_contour_geometry.centroid.coords[0]
    import numpy as np

    angle_rad = np.radians(angle_degrees)
    dx, dy = np.cos(angle_rad), np.sin(angle_rad)
    diag = max(
        original_contour.bounds[2] - original_contour.bounds[0],
        original_contour.bounds[3] - original_contour.bounds[1],
    ) * WallSettings.CONNECTION_RAY_DIAGONAL_MULTIPLIER
    ray = LineString([(cx, cy), (cx + dx * diag, cy + dy * diag)])

    offset_hit = ray.intersection(offset_contour_geometry.boundary)
    if offset_hit.is_empty:
        start_point = Point(cx, cy)
    elif offset_hit.geom_type == "Point":
        start_point = offset_hit
    elif offset_hit.geom_type == "MultiPoint":
        start_point = min(offset_hit.geoms, key=lambda p: p.distance(Point(cx, cy)))
    else:
        start_point = Point(cx, cy)

    original_hit = ray.intersection(original_contour.boundary)
    if original_hit.is_empty:
        end_point = start_point
    elif original_hit.geom_type == "Point":
        end_point = original_hit
    elif original_hit.geom_type == "MultiPoint":
        end_point = min(original_hit.geoms, key=lambda p: p.distance(start_point))
    else:
        end_point = start_point

    return LineString([start_point, end_point]), start_point



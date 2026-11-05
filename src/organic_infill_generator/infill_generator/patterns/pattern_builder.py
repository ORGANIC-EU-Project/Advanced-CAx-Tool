"""
Pattern request resolution and toolpath assembly helpers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, Point, base
from shapely.ops import linemerge, substring

from ..models import PatternRequest
from ...common.geometry import reorder_ring
from .pattern_resolution import (
    density_to_infill_spacing,
    derive_defaults_from_line_distance,
    estimate_density_percent,
    line_distance_from_geometry,
    pattern_distance_factor_base,
    pattern_step_ratio,
)
from ..settings import (
    ApiDefaults,
    GeometrySettings,
    PatternCatalog,
    WallSettings,
)

@dataclass
class ResolvedPatternRequest:
    """
    Internal resolved request used only between the resolver and the generator.
    """
    num_walls: int
    line_width: float
    wall_spacing: float
    fillet_radius: float
    infill_density: float
    infill_pattern: str
    extrude_height: float
    layer_height: float
    bottom_layers: int
    top_layers: int
    base_line_distance: float
    line_distance: float
    pattern_distance_factor: float
    infill_spacing: float
    pattern_step: float
    pattern_amplitude: float
    pattern_sequence: list[str]
    estimated_density: float


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))

def _geometry_to_coord_lists(geometry: base.BaseGeometry) -> list[list[tuple[float, float]]]:
    if geometry.is_empty:
        return []
    if geometry.geom_type in {"LineString", "LinearRing"}:
        return [list(geometry.coords)]
    if geometry.geom_type == "MultiLineString":
        return [list(line.coords) for line in geometry.geoms]
    if geometry.geom_type == "GeometryCollection":
        result: list[list[tuple[float, float]]] = []
        for sub_geometry in geometry.geoms:
            result.extend(_geometry_to_coord_lists(sub_geometry))
        return result
    return []


def _primary_boundary_line(geometry: base.BaseGeometry) -> LineString:
    if geometry.geom_type == "Polygon":
        return LineString(list(geometry.boundary.coords))
    if geometry.geom_type == "MultiPolygon":
        polygon = max(geometry.geoms, key=lambda item: item.area)
        return LineString(list(polygon.boundary.coords))
    if geometry.geom_type in {"LineString", "LinearRing"}:
        return LineString(list(geometry.coords))
    raise ValueError(f"Unsupported wall geometry type: {geometry.geom_type}")


def _build_diagonal_wall_connector(
        *,
        current_point: tuple[float, float],
        wall_line: LineString,
        connection_angle_degrees: float,
) -> tuple[list[tuple[float, float]], LineString]:
    """Connect infill to a wall with an angled seam, then reorder that wall."""
    projected_distance = wall_line.project(Point(current_point))
    projected_point = wall_line.interpolate(projected_distance)

    radial_gap = max(Point(current_point).distance(projected_point), GeometrySettings.COORD_EPSILON)
    clamped_angle = _clamp(connection_angle_degrees, -89.0, 89.0)
    angle_radians = math.radians(clamped_angle)
    tangential_advance = radial_gap * math.tan(angle_radians)

    if abs(tangential_advance) > GeometrySettings.COORD_EPSILON and wall_line.length > GeometrySettings.COORD_EPSILON:
        projected_distance = (projected_distance + tangential_advance) % wall_line.length
        projected_point = wall_line.interpolate(projected_distance)

    connector_end = (float(projected_point.x), float(projected_point.y))
    connector = [current_point, connector_end]
    reordered = reorder_ring(wall_line, projected_point)
    return connector, reordered


def _collect_intersection_y_values(geometry: base.BaseGeometry) -> list[float]:
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Point":
        return [float(geometry.y)]
    if geometry.geom_type == "MultiPoint":
        return [float(item.y) for item in geometry.geoms]
    if geometry.geom_type in {"LineString", "LinearRing"}:
        return [float(coord[1]) for coord in geometry.coords]
    if geometry.geom_type in {"MultiLineString", "GeometryCollection"}:
        values: list[float] = []
        for item in geometry.geoms:
            values.extend(_collect_intersection_y_values(item))
        return values
    return []


def _find_vertical_extension_target(
        *,
        point: tuple[float, float],
        neighbor: tuple[float, float],
        wall_geometry: base.BaseGeometry,
        wall_gap_length: float,
) -> tuple[float, float] | None:
    """Extend the first infill segment toward the innermost wall when possible.

    Disabled by ``WallSettings.EXTEND_INFILL_START_TO_WALL``. The probe below is
    vertical, so the gap it leaves is a vertical one: against a wall at angle
    theta the perpendicular clearance is only ``wall_gap_length * cos(theta)``,
    which is why an octagon chamfer ended up at 2.62 mm and an ellipse at 1.39
    where the intent was 3.70.
    """
    if not WallSettings.EXTEND_INFILL_START_TO_WALL:
        return None
    delta_y = float(neighbor[1] - point[1])
    if abs(delta_y) <= GeometrySettings.COORD_EPSILON:
        return None

    direction = -1.0 if delta_y > 0.0 else 1.0

    wall_line = _primary_boundary_line(wall_geometry)
    minx, miny, maxx, maxy = wall_line.bounds
    probe_margin = max(wall_gap_length, 1.0)
    vertical_probe = LineString([
        (float(point[0]), float(miny - probe_margin)),
        (float(point[0]), float(maxy + probe_margin)),
    ])
    intersection = wall_line.intersection(vertical_probe)
    y_values = _collect_intersection_y_values(intersection)
    if not y_values:
        return None

    candidates = [
        y_value
        for y_value in y_values
        if direction * (y_value - float(point[1])) > GeometrySettings.COORD_EPSILON
    ]
    if not candidates:
        return None

    wall_y = min(candidates) if direction > 0.0 else max(candidates)
    target_y = wall_y - direction * wall_gap_length
    if direction * (target_y - float(point[1])) <= GeometrySettings.COORD_EPSILON:
        return None
    return float(point[0]), float(target_y)


def _segments_are_forward_collinear(
        first_point: tuple[float, float],
        middle_point: tuple[float, float],
        last_point: tuple[float, float],
        *,
        angle_tolerance: float = 1e-3,
        cos_tolerance: float = 0.99,
) -> bool:
    """Return true when two consecutive path segments continue in one direction."""
    a = np.asarray(middle_point, dtype=float) - np.asarray(first_point, dtype=float)
    b = np.asarray(last_point, dtype=float) - np.asarray(middle_point, dtype=float)
    len_a = float(np.linalg.norm(a))
    len_b = float(np.linalg.norm(b))
    if len_a <= GeometrySettings.LENGTH_EPSILON or len_b <= GeometrySettings.LENGTH_EPSILON:
        return False

    sin_angle = float((a[0] * b[1] - a[1] * b[0]) / (len_a * len_b))
    cos_angle = float(np.dot(a, b) / (len_a * len_b))
    return abs(sin_angle) < angle_tolerance and cos_angle > cos_tolerance


def _trim_wall_ring_with_gap(
        wall_line: LineString,
        wall_gap_length: float,
) -> list[tuple[float, float]]:
    """
    Trim a reordered closed wall ring by leaving a seam gap at the tail of the
    traversed loop.
    """
    if wall_gap_length <= GeometrySettings.COORD_EPSILON:
        return list(wall_line.coords)

    cut_length = wall_line.length - wall_gap_length
    if cut_length <= wall_gap_length:
        return list(wall_line.coords)

    return list(substring(wall_line, 0.0, cut_length).coords)


def _reverse_ring_preserving_start(wall_line: LineString) -> LineString:
    coords = list(wall_line.coords)
    if len(coords) < 3:
        return wall_line

    if np.allclose(coords[0], coords[-1], atol=GeometrySettings.COORD_EPSILON):
        unique_coords = coords[:-1]
        reversed_coords = [unique_coords[0], *reversed(unique_coords[1:]), unique_coords[0]]
    else:
        reversed_coords = [coords[0], *reversed(coords[1:])]

    return LineString(reversed_coords)


def resolve_pattern_request(payload: dict | PatternRequest) -> ResolvedPatternRequest:
    """
    Resolve a pattern recipe into stable internal parameters.
    """
    if isinstance(payload, PatternRequest):
        public_request = payload
    elif isinstance(payload, dict):
        requested_fillet = payload.get("fillet_radius", ApiDefaults.FILLET_RADIUS)
        line_width = float(payload.get("line_width", ApiDefaults.LINE_WIDTH))
        if requested_fillet is None:
            fillet_radius = line_width
        else:
            fillet_radius = max(float(requested_fillet), 0.0)

        spacing_ratio = payload.get("wall_spacing_ratio", WallSettings.SPACING_RATIO)
        public_request = PatternRequest(
            num_walls=int(payload.get("num_walls", ApiDefaults.NUM_WALLS)),
            line_width=line_width,
            wall_spacing=float(payload.get("wall_spacing", WallSettings.SPACING)),
            wall_spacing_ratio=None if spacing_ratio is None else float(spacing_ratio),
            fillet_radius=fillet_radius,
            infill_density=float(payload.get("infill_density", ApiDefaults.INFILL_DENSITY)),
            infill_pattern=str(payload.get("infill_pattern", ApiDefaults.INFILL_PATTERN)),
            extrude_height=float(payload.get("extrude_height", ApiDefaults.EXTRUDE_HEIGHT)),
            layer_height=float(payload.get("layer_height", ApiDefaults.LAYER_HEIGHT)),
            bottom_layers=int(payload.get("bottom_layers", ApiDefaults.BOTTOM_LAYERS)),
            top_layers=int(payload.get("top_layers", ApiDefaults.TOP_LAYERS)),
        )
    else:
        raise TypeError("generate_pattern expects a dict payload or PatternRequest.")

    num_walls = int(public_request.num_walls)
    line_width = float(public_request.line_width)
    # A ratio, when given, wins over the absolute value. See PatternRequest.
    wall_spacing = float(public_request.wall_spacing_mm)
    wall_spacing_ratio = (None if public_request.wall_spacing_ratio is None
                          else float(public_request.wall_spacing_ratio))
    infill_density = float(public_request.infill_density)
    infill_pattern = str(public_request.infill_pattern)
    fillet_radius = float(public_request.fillet_radius)
    extrude_height = float(public_request.extrude_height)
    layer_height = float(public_request.layer_height)
    bottom_layers = int(public_request.bottom_layers)
    top_layers = int(public_request.top_layers)

    if line_width <= 0:
        raise ValueError("'line_width' must be > 0.")
    if num_walls < 0:
        raise ValueError("'num_walls' must be >= 0.")
    if wall_spacing_ratio is not None and wall_spacing_ratio <= -1.0:
        raise ValueError("'wall_spacing_ratio' must be greater than -1.0 "
                         "(at -1.0 adjacent centrelines would coincide).")
    if wall_spacing <= -line_width:
        raise ValueError("'wall_spacing' must be greater than -line_width.")
    if extrude_height < 0:
        raise ValueError("'extrude_height' must be >= 0.")
    if layer_height <= 0:
        raise ValueError("'layer_height' must be > 0.")
    if bottom_layers < 0:
        raise ValueError("'bottom_layers' must be >= 0.")
    if top_layers < 0:
        raise ValueError("'top_layers' must be >= 0.")

    solid_bottom_height = float(bottom_layers * layer_height)
    solid_top_height = float(top_layers * layer_height)
    if solid_bottom_height + solid_top_height - extrude_height > GeometrySettings.AREA_EPSILON:
        raise ValueError("Bottom/top solid layers exceed the requested 'extrude_height'.")

    sequence = PatternCatalog.PUBLIC_PATTERN_MAP.get(infill_pattern)
    pattern_sequence = list(sequence) if sequence is not None else None
    if pattern_sequence is None:
        valid = ", ".join(sorted(PatternCatalog.PUBLIC_PATTERN_MAP.keys()))
        raise ValueError(f"Unknown infill_pattern '{infill_pattern}'. Valid values: {valid}")
    if not pattern_sequence:
        raise ValueError(f"Resolved empty pattern sequence for '{infill_pattern}'.")

    step_ratio = pattern_step_ratio(infill_pattern, pattern_sequence)
    pattern_distance_factor = pattern_distance_factor_base(infill_pattern, pattern_sequence)

    base_line_distance = density_to_infill_spacing(
        line_width=line_width,
        density_percent=infill_density,
    )
    target_line_distance = base_line_distance * pattern_distance_factor
    derived_infill_spacing, derived_pattern_amplitude, derived_pattern_step = derive_defaults_from_line_distance(
        line_distance=target_line_distance,
        step_ratio=step_ratio,
        pattern_sequence=pattern_sequence,
        line_width=line_width,
        wall_spacing=wall_spacing,
    )
    infill_spacing = float(derived_infill_spacing)
    pattern_amplitude = float(derived_pattern_amplitude)
    pattern_step = float(derived_pattern_step)

    line_distance = line_distance_from_geometry(
        pattern_sequence=pattern_sequence,
        infill_spacing=infill_spacing,
        pattern_amplitude=pattern_amplitude,
        pattern_step=pattern_step,
    )

    if infill_spacing <= 0:
        raise ValueError("Resolved 'infill_spacing' must be > 0.")
    if pattern_step <= 0:
        raise ValueError("Resolved 'pattern_step' must be > 0.")
    if pattern_amplitude < 0:
        raise ValueError("Resolved 'pattern_amplitude' must be >= 0.")

    estimated_density = estimate_density_percent(
        line_width=line_width,
        line_distance=line_distance,
        pattern_distance_factor=pattern_distance_factor,
    )

    return ResolvedPatternRequest(
        num_walls=num_walls,
        line_width=line_width,
        wall_spacing=wall_spacing,
        fillet_radius=fillet_radius,
        infill_density=infill_density,
        infill_pattern=infill_pattern,
        extrude_height=extrude_height,
        layer_height=layer_height,
        bottom_layers=bottom_layers,
        top_layers=top_layers,
        base_line_distance=base_line_distance,
        line_distance=line_distance,
        pattern_distance_factor=pattern_distance_factor,
        infill_spacing=infill_spacing,
        pattern_step=pattern_step,
        pattern_amplitude=pattern_amplitude,
        pattern_sequence=pattern_sequence,
        estimated_density=estimated_density,
    )


def _build_full_toolpath(
        infill_path_points: list[tuple[float, float]],
        infill_path_geometry: base.BaseGeometry,
        wall_geometries: list[base.BaseGeometry],
        wall_gap_length: float,
        wall_connection_angle_degrees: float,
) -> tuple[np.ndarray, base.BaseGeometry, tuple[float, float] | None, tuple[float, float] | None, list[int]]:
    """
    Assemble infill columns, optional wall loops, and seam connectors.

    The returned protected indices mark vertices that should not be rounded
    later, usually because they are deliberate wall-entry or seam points.
    """
    path_parts = _geometry_to_coord_lists(infill_path_geometry)
    full_coords: list[tuple[float, float]] = list(infill_path_points)
    wall_plot_parts: list[tuple[list[tuple[float, float]], list[tuple[float, float]]]] = []
    infill_extension_parts: list[list[tuple[float, float]]] = []
    fillet_protected_indices: set[int] = set()

    if wall_geometries and len(full_coords) >= 2:
        innermost_wall = wall_geometries[-1]
        # Pull the first infill point toward the inner wall along its initial
        # vertical direction. This avoids leaving a short unprinted gap before
        # the first wall connector is added.
        start_extension = _find_vertical_extension_target(
            point=full_coords[0],
            neighbor=full_coords[1],
            wall_geometry=innermost_wall,
            wall_gap_length=wall_gap_length,
        )
        if start_extension is not None:
            full_coords = [start_extension, *full_coords]
            infill_extension_parts.append([start_extension, tuple(infill_path_points[0])])

    last_point = full_coords[-1] if full_coords else None

    wall_indices = list(range(len(wall_geometries) - 1, -1, -1))

    for wall_position, wall_index in enumerate(wall_indices):
        wall_line = _primary_boundary_line(wall_geometries[wall_index])

        if last_point is None:
            wall_coords = _trim_wall_ring_with_gap(
                wall_line,
                wall_gap_length=wall_gap_length,
            )
            wall_plot_parts.append(([], wall_coords))
            full_coords.extend(wall_coords)
            if wall_coords:
                last_point = wall_coords[-1]
            continue

        effective_connection_angle = float(wall_connection_angle_degrees)
        if WallSettings.SWAP_GAP_SIDE_WITH_INFILL_DIRECTION:
            effective_connection_angle = -effective_connection_angle

        # Enter each wall at a controlled angle and then traverse the wall as a
        # nearly closed loop. The missing tail is the seam gap.
        connector, reordered = _build_diagonal_wall_connector(
            current_point=last_point,
            wall_line=wall_line,
            connection_angle_degrees=effective_connection_angle,
        )
        if WallSettings.SWAP_GAP_SIDE_WITH_INFILL_DIRECTION:
            reordered = _reverse_ring_preserving_start(reordered)

        wall_coords = _trim_wall_ring_with_gap(
            reordered,
            wall_gap_length=wall_gap_length,
        )

        connector = connector if wall_coords else []
        wall_plot_parts.append((connector, wall_coords))

        if wall_coords:
            if len(full_coords) > 0:
                fillet_protected_indices.add(len(full_coords) - 1)
            wall_entry = tuple(map(float, wall_coords[0]))
            if len(full_coords) >= 2 and _segments_are_forward_collinear(
                    full_coords[-2],
                    full_coords[-1],
                    wall_entry,
            ):
                full_coords[-1] = wall_entry
            elif (
                    not full_coords
                    or np.linalg.norm(np.asarray(wall_entry, dtype=float) - np.asarray(full_coords[-1], dtype=float))
                    > GeometrySettings.COORD_EPSILON
            ):
                full_coords.append(wall_entry)
            if len(full_coords) > 0:
                fillet_protected_indices.add(len(full_coords) - 1)
            full_coords.extend(wall_coords[1:])
            last_point = wall_coords[-1]

    plot_lines = []
    for extension_part in infill_extension_parts:
        if len(extension_part) >= 2:
            plot_lines.append(LineString(extension_part))
    for part in path_parts:
        if len(part) >= 2:
            plot_lines.append(LineString(part))
    for connector, wall_coords in wall_plot_parts:
        if len(connector) >= 2:
            plot_lines.append(LineString(connector))
        if len(wall_coords) >= 2:
            plot_lines.append(LineString(wall_coords))

    full_path_geometry = linemerge(plot_lines) if plot_lines else LineString()
    full_path = np.array(full_coords, dtype=float) if full_coords else np.zeros((0, 2))

    if len(full_path) >= 2:
        deduped = [full_path[0]]
        for point in full_path[1:]:
            if np.linalg.norm(point - deduped[-1]) > GeometrySettings.COORD_EPSILON:
                deduped.append(point)
        full_path = np.asarray(deduped, dtype=float)

    start_point = tuple(full_path[0]) if len(full_path) else None
    end_point = tuple(full_path[-1]) if len(full_path) else None
    protected = sorted(index for index in fillet_protected_indices if 0 <= index < len(full_path))
    return full_path, full_path_geometry, start_point, end_point, protected

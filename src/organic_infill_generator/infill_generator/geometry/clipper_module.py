"""
clipper_module.py
=================
Eulerian path assembly from precomputed infill columns.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point, Polygon, base
from shapely.ops import linemerge, substring, unary_union

from ..settings import GeometrySettings, PathSettings
from .snake_repair import repair_snake_overlaps


def clip_shape_to_contour(
        pattern_geometry: base.BaseGeometry,
        contour_geometry: base.BaseGeometry,
) -> base.BaseGeometry:
    """Clip a pattern geometry to the printable infill contour."""
    if pattern_geometry.is_empty:
        return pattern_geometry
    return pattern_geometry.intersection(contour_geometry)


def longest_linestring(geometry: base.BaseGeometry) -> LineString | None:
    """Return the longest line component from a shapely geometry."""
    if geometry.is_empty:
        return None
    if isinstance(geometry, LineString):
        return geometry
    if geometry.geom_type == "MultiLineString":
        return max(geometry.geoms, key=lambda line: line.length)
    if geometry.geom_type == "GeometryCollection":
        lines = [candidate for candidate in geometry.geoms if isinstance(candidate, LineString)]
        if not lines:
            return None
        return max(lines, key=lambda line: line.length)
    return None


def boundary_shortest_arc(
        boundary: LineString | base.BaseGeometry,
        start_point: Point,
        end_point: Point,
) -> base.BaseGeometry:
    """Return the shorter path along a closed boundary between two points."""
    boundary = LineString(boundary)
    total_length = boundary.length
    start_distance = boundary.project(start_point)
    end_distance = boundary.project(end_point)

    if abs(start_distance - end_distance) < GeometrySettings.AREA_EPSILON:
        return Point(start_point)

    def _sub(a: float, b: float) -> LineString:
        return substring(boundary, a, b, normalized=False)

    def _ordered_lines(*parts: LineString) -> base.BaseGeometry:
        non_empty = [
            part for part in parts
            if isinstance(part, LineString) and part.length > GeometrySettings.LENGTH_EPSILON
        ]
        if not non_empty:
            return GeometryCollection()
        if len(non_empty) == 1:
            return non_empty[0]
        merged = linemerge(MultiLineString(non_empty))
        return merged if not merged.is_empty else MultiLineString(non_empty)

    if start_distance <= end_distance:
        forward = _sub(start_distance, end_distance)
    else:
        forward = _ordered_lines(_sub(start_distance, total_length), _sub(0.0, end_distance))

    if end_distance <= start_distance:
        backward = _sub(end_distance, start_distance)
    else:
        backward = _ordered_lines(_sub(end_distance, total_length), _sub(0.0, start_distance))

    return forward if forward.length <= backward.length else backward


def _sorted_segment_endpoints(segment: LineString) -> tuple[Point, Point]:
    coords = list(segment.coords)
    start_point = Point(coords[0])
    end_point = Point(coords[-1])
    endpoints = [start_point, end_point]
    endpoints.sort(key=lambda point: (point.y, point.x))
    return endpoints[0], endpoints[1]


def _orient_segment(segment: LineString, start_point: Point) -> LineString:
    coords = list(segment.coords)
    if Point(coords[0]).distance(start_point) <= Point(coords[-1]).distance(start_point):
        return segment
    return LineString(coords[::-1])


def _trim_segment(segment: LineString, trim_length: float) -> LineString:
    if trim_length <= GeometrySettings.LENGTH_EPSILON or segment.length <= 2.0 * trim_length:
        return segment
    trimmed = substring(segment, trim_length, segment.length - trim_length, normalized=False)
    if isinstance(trimmed, LineString) and trimmed.length > GeometrySettings.LENGTH_EPSILON:
        return trimmed
    return segment


def _flatten_lines(geometry: base.BaseGeometry) -> list[LineString]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, LineString):
        return [geometry]
    if geometry.geom_type == "MultiLineString":
        return list(geometry.geoms)
    if geometry.geom_type == "GeometryCollection":
        result: list[LineString] = []
        for sub_geometry in geometry.geoms:
            result.extend(_flatten_lines(sub_geometry))
        return result
    return []


def _ordered_connector_path(
        pieces: list[base.BaseGeometry],
        start_point: Point,
) -> LineString | GeometryCollection:
    """Order connector fragments by nearest endpoint so they form one path."""
    lines: list[LineString] = []
    for piece in pieces:
        lines.extend(_flatten_lines(piece))

    lines = [line for line in lines if line.length > GeometrySettings.LENGTH_EPSILON]
    if not lines:
        return GeometryCollection()

    remaining = lines[:]
    current = np.array([start_point.x, start_point.y], dtype=float)
    ordered_coords: list[tuple[float, float]] = [(start_point.x, start_point.y)]

    # Shapely clipping can return connector pieces in arbitrary order. Greedy
    # nearest-end ordering is enough here because every connector is short and
    # lies on the same boundary route.
    while remaining:
        best_index = -1
        best_reverse = False
        best_distance = float("inf")

        for index, line in enumerate(remaining):
            coords = list(line.coords)
            start_coords = np.asarray(coords[0], dtype=float)
            end_coords = np.asarray(coords[-1], dtype=float)
            start_distance = float(np.linalg.norm(start_coords - current))
            end_distance = float(np.linalg.norm(end_coords - current))

            if start_distance < best_distance:
                best_index = index
                best_reverse = False
                best_distance = start_distance
            if end_distance < best_distance:
                best_index = index
                best_reverse = True
                best_distance = end_distance

        if best_index < 0:
            break

        coords = list(remaining.pop(best_index).coords)
        if best_reverse:
            coords = coords[::-1]

        if ordered_coords:
            first = np.asarray(coords[0], dtype=float)
            if np.linalg.norm(first - np.asarray(ordered_coords[-1], dtype=float)) <= GeometrySettings.COORD_EPSILON:
                coords = coords[1:]

        ordered_coords.extend(coords)
        if ordered_coords:
            current = np.asarray(ordered_coords[-1], dtype=float)

    if len(ordered_coords) < 2:
        return GeometryCollection()
    return LineString(ordered_coords)


def _connect_points_inside_boundary(
        infill_boundary: base.BaseGeometry,
        boundary: LineString | base.BaseGeometry,
        start_point: Point,
        end_point: Point,
        start_boundary_point: Point,
        end_boundary_point: Point,
) -> base.BaseGeometry:
    """Connect two clipped column fragments through the infill boundary."""
    pieces: list[base.BaseGeometry] = []
    spoke_out = LineString([(start_point.x, start_point.y), (start_boundary_point.x, start_boundary_point.y)])
    if spoke_out.length > GeometrySettings.LENGTH_EPSILON:
        pieces.append(spoke_out)

    arc = boundary_shortest_arc(boundary, start_boundary_point, end_boundary_point)
    if not arc.is_empty and arc.length > GeometrySettings.LENGTH_EPSILON:
        pieces.append(arc)

    spoke_in = LineString([(end_boundary_point.x, end_boundary_point.y), (end_point.x, end_point.y)])
    if spoke_in.length > GeometrySettings.LENGTH_EPSILON:
        pieces.append(spoke_in)

    if not pieces:
        connector = LineString([(start_point.x, start_point.y), (end_point.x, end_point.y)])
        if connector.length > GeometrySettings.LENGTH_EPSILON:
            padded_boundary = infill_boundary.buffer(PathSettings.CONNECTOR_COVER_TOLERANCE)
            if connector.covered_by(padded_boundary):
                return connector
        return GeometryCollection()
    return _ordered_connector_path(pieces, start_point=start_point)


def _direct_connector_inside_boundary(
        infill_boundary: base.BaseGeometry,
        start_point: Point,
        end_point: Point,
) -> LineString | None:
    connector = LineString([(start_point.x, start_point.y), (end_point.x, end_point.y)])
    if connector.length <= GeometrySettings.LENGTH_EPSILON:
        return None
    padded_boundary = infill_boundary.buffer(PathSettings.CONNECTOR_COVER_TOLERANCE)
    return connector if connector.covered_by(padded_boundary) else None


def _fragment_connector(
        infill_boundary: base.BaseGeometry,
        boundary: LineString | base.BaseGeometry,
        start_point: Point,
        end_point: Point,
) -> base.BaseGeometry:
    """
    Connect two column fragments without duplicating a wall-like boundary path.

    Boundary arcs are still allowed because they are useful in concave shapes.
    When the direct pattern connector and the boundary arc are effectively the
    same job, keep only the direct connector so the output cannot contain both
    a wall-following segment and a near-identical pattern segment.
    """
    direct = _direct_connector_inside_boundary(infill_boundary, start_point, end_point)
    arc = boundary_shortest_arc(boundary, start_point, end_point)

    if direct is not None:
        return direct
    if not arc.is_empty and arc.length > GeometrySettings.LENGTH_EPSILON:
        return arc
    if direct is not None:
        return direct
    return GeometryCollection()


def clip_and_reconnect_columns(
        raw_columns: list[LineString],
        infill_boundary: base.BaseGeometry,
        line_width: float = 0.0,
        allow_boundary_reconnect: bool = True,
) -> list[LineString]:
    """
    Clip a list of raw (unclipped) pattern columns to the infill boundary,
    reconnecting any fragments produced by the clip via the shortest boundary arc.

    This is the single entry point between sequence_drawer (pattern generation)
    and assemble_eulerian_path (snake assembly).
    """
    result: list[LineString] = []
    clipping_boundary = infill_boundary
    if line_width > GeometrySettings.AREA_EPSILON:
        safe_margin = float(line_width) / 2.0
        eroded = infill_boundary.buffer(-safe_margin)
        if eroded is not None and not eroded.is_empty:
            clipping_boundary = eroded

    for col in sorted(raw_columns, key=lambda c: c.centroid.x):
        clipped = clip_shape_to_contour(col, clipping_boundary)
        if clipped.is_empty:
            continue
        reconnected = reconnect_column_fragments(
            clipped,
            clipping_boundary,
            allow_boundary_reconnect=allow_boundary_reconnect,
        )
        if reconnected is not None:
            result.append(reconnected)
    return result


def reconnect_column_fragments(
        clipped_geometry: base.BaseGeometry,
        infill_boundary: base.BaseGeometry,
        *,
        allow_boundary_reconnect: bool = True,
) -> LineString | None:
    """
    When a raw column is clipped into multiple fragments by the contour boundary,
    reconnect them bottom-to-top via the shortest boundary arc between consecutive
    fragment endpoints -- instead of discarding all fragments but the longest.

    The path is always built ascending (low Y -> high Y).  assemble_eulerian_path
    will flip it for odd-indexed columns via _orient_segment, exactly as it does
    for any other column.

    Parameters
    ----------
    clipped_geometry : result of clip_shape_to_contour for a single raw column.
    infill_boundary  : the enclosing polygon (used to walk its boundary).
    """
    segments = _flatten_lines(clipped_geometry)
    segments = [s for s in segments if s.length > GeometrySettings.LENGTH_EPSILON]
    if not segments:
        return None
    if len(segments) == 1:
        return segments[0]
    if not allow_boundary_reconnect:
        return max(segments, key=lambda line: line.length)

    # Always sort ascending; assemble_eulerian_path handles the final direction.
    segments.sort(key=lambda s: float(np.array(s.coords)[:, 1].mean()))

    boundary = infill_boundary.boundary

    all_coords: list[tuple[float, float]] = []
    for seg in segments:
        lo, _ = _sorted_segment_endpoints(seg)
        oriented = _orient_segment(seg, lo)          # start each fragment at its bottom
        seg_coords = list(oriented.coords)

        if all_coords:
            prev_end = Point(all_coords[-1])
            this_start = Point(seg_coords[0])

            connector = _fragment_connector(
                infill_boundary,
                boundary,
                prev_end,
                this_start,
            )
            for connector_line in _flatten_lines(connector):
                connector_coords = list(connector_line.coords)
                if (
                    connector_coords
                    and Point(connector_coords[0]).distance(Point(all_coords[-1]))
                    > Point(connector_coords[-1]).distance(Point(all_coords[-1]))
                ):
                    connector_coords = connector_coords[::-1]
                if (
                    connector_coords
                    and Point(connector_coords[0]).distance(Point(all_coords[-1])) <= GeometrySettings.COORD_EPSILON
                ):
                    connector_coords = connector_coords[1:]
                all_coords.extend(connector_coords)

            if all_coords and Point(seg_coords[0]).distance(Point(all_coords[-1])) > GeometrySettings.COORD_EPSILON:
                return max(segments, key=lambda line: line.length)
            seg_coords = seg_coords[1:]

        all_coords.extend(seg_coords)

    return LineString(all_coords) if len(all_coords) >= 2 else None


@dataclass(frozen=True)
class SnakePart:
    """One piece of the assembled snake, with the columns a connector joins.

    ``assemble_eulerian_path`` unions the connectors before returning them,
    which loses which two columns each one bridges -- and that pairing is what
    a trimming rule needs, because the margin has to be measured against those
    two columns and not against every column in the layer.
    """

    kind: str                      # "column" | "connector"
    geometry: base.BaseGeometry
    index: int                     # column index, or left column for a connector
    right: int | None = None       # right column index; connectors only


def assemble_eulerian_path(
        infill_boundary: base.BaseGeometry,
        columns: list[LineString],
        line_width: float = 0.0,
        wall_spacing: float = 0.0,
        *,
        with_parts: bool = False,
) -> tuple[
    base.BaseGeometry,
    base.BaseGeometry,
    base.BaseGeometry,
    tuple[float, float] | None,
    list[tuple[float, float]],
]:
    """
    Assemble a continuous snake path from definitive pre-clipped columns.

    With ``with_parts=True`` a sixth element is appended: the ordered list of
    :class:`SnakePart`, so callers can tell a column from a connector and know
    which two columns each connector bridges. The five-element return is
    unchanged, so existing callers keep working.

    ``wall_spacing`` is the resolved gap in model units. Together with
    ``line_width`` it sets the band inside which a connector counts as running
    on top of a column -- see :mod:`.snake_repair`. The bead is one
    ``line_width`` wide and centred on the path, so two parallel centrelines
    that far apart touch edge to edge; closer than that they overlap. A negative
    spacing means the beads are meant to overlap, and the band narrows to match.
    """
    candidates = [
        column
        for column in columns
        if column is not None and column.length > GeometrySettings.LENGTH_EPSILON
    ]

    if not candidates:
        empty = (GeometryCollection(), GeometryCollection(), GeometryCollection(), None, [])
        return (*empty, []) if with_parts else empty

    candidates.sort(key=lambda geometry: geometry.centroid.x)
    # The forbidden area is the bead of every column already accepted, kept
    # as separate pieces. A column is only cut by the pieces whose bounds
    # reach it: a bead that does not touch the column cannot remove any of
    # it. Differencing against the running union of ALL earlier beads made
    # the loop quadratic -- MEASURED on honeycomb d90 / 3000 mm deep / line
    # width 7: 0.3 / 0.8 / 1.9 / 6.1 / 22.5 s for 500 / 1000 / 2000 / 4000 /
    # 8000 mm wide (26 -> 419 columns), most of it in difference/union.
    forbidden_pieces: list[base.BaseGeometry] = []
    forbidden_bounds: list[tuple[float, float, float, float]] = []
    # Eroded interior: boundary arcs lie exactly on the polygon boundary, so
    # shrinking by a tiny epsilon excludes them from the forbidden-area buffer.
    # This prevents reconnection arcs from blocking adjacent columns.
    interior = infill_boundary.buffer(-GeometrySettings.LENGTH_EPSILON)
    final_columns = []

    for segment in candidates:
        if line_width > GeometrySettings.AREA_EPSILON and forbidden_pieces:
            min_x, min_y, max_x, max_y = segment.bounds
            nearby = [
                piece
                for piece, (piece_min_x, piece_min_y, piece_max_x, piece_max_y)
                in zip(forbidden_pieces, forbidden_bounds)
                if piece_min_x <= max_x and piece_max_x >= min_x
                and piece_min_y <= max_y and piece_max_y >= min_y
            ]
            if nearby:
                forbidden_area = nearby[0] if len(nearby) == 1 else unary_union(nearby)
                remaining_segment = segment.difference(forbidden_area)
                remaining_lines = _flatten_lines(remaining_segment)
                if len(remaining_lines) == 1:
                    segment = remaining_lines[0]
                elif len(remaining_lines) == 0:
                    segment = None
        if segment is None or segment.length < GeometrySettings.LENGTH_EPSILON:
            continue
        if line_width > GeometrySettings.AREA_EPSILON:
            interior_part = segment.intersection(interior)
            if not interior_part.is_empty:
                piece = interior_part.buffer(line_width / 2.0, cap_style=2)
                if not piece.is_empty:
                    forbidden_pieces.append(piece)
                    forbidden_bounds.append(piece.bounds)
        final_columns.append(segment)

    trim_length = max(line_width * PathSettings.COLUMN_END_TRIM_FACTOR, 0.0)
    info: list[dict] = []
    for segment in final_columns:
        lower_endpoint, upper_endpoint = _sorted_segment_endpoints(segment)
        trimmed_segment = _trim_segment(segment, trim_length=trim_length)
        trimmed_coords = list(trimmed_segment.coords)
        info.append({
            "geometry": segment,
            "trimmed_geometry": trimmed_segment,
            "lower_endpoint": lower_endpoint,
            "upper_endpoint": upper_endpoint,
            "trimmed_start": Point(trimmed_coords[0]),
            "trimmed_end": Point(trimmed_coords[-1]),
            "center_x": float(np.asarray(segment.coords)[:, 0].mean()),
        })

    if not info:
        return GeometryCollection(), GeometryCollection(), GeometryCollection(), None, []

    oriented_info: list[dict] = []
    for index, column_info in enumerate(info):
        desired_start = column_info["lower_endpoint"] if index % 2 == 0 else column_info["upper_endpoint"]
        desired_end = column_info["upper_endpoint"] if index % 2 == 0 else column_info["lower_endpoint"]
        oriented_segment = _orient_segment(column_info["geometry"], desired_start)
        trimmed_segment = _trim_segment(oriented_segment, trim_length=trim_length)
        trimmed_coords = list(trimmed_segment.coords)
        oriented_info.append({
            "geometry": oriented_segment,
            "trimmed_geometry": trimmed_segment,
            "start_boundary_point": Point(list(oriented_segment.coords)[0]),
            "end_boundary_point": Point(list(oriented_segment.coords)[-1]),
            "start_point": Point(trimmed_coords[0]),
            "end_point": Point(trimmed_coords[-1]),
            "desired_end": desired_end,
        })

    start_point = (oriented_info[0]["start_point"].x, oriented_info[0]["start_point"].y)
    connectors: list[base.BaseGeometry] = []
    boundary = infill_boundary.boundary
    if boundary.geom_type not in {"LineString", "LinearRing"}:
        raise ValueError(PathSettings.HOLES_NOT_SUPPORTED_MESSAGE)

    # One slot per gap, holding None when no connector was produced. Appending
    # only the non-empty ones compacts the list, and the assembly below indexes
    # it by column position -- one skipped connector would silently shift every
    # later pairing.
    connector_slots: list[base.BaseGeometry | None] = [None] * max(len(oriented_info) - 1, 0)
    for index in range(len(oriented_info) - 1):
        left = oriented_info[index]
        right = oriented_info[index + 1]
        connector = _connect_points_inside_boundary(
            infill_boundary=infill_boundary,
            boundary=boundary,
            start_point=left["end_point"],
            end_point=right["start_point"],
            start_boundary_point=left["end_boundary_point"],
            end_boundary_point=right["start_boundary_point"],
        )
        if not connector.is_empty and connector.length > GeometrySettings.LENGTH_EPSILON:
            connector_slots[index] = connector

    # An oblique or curved edge makes the connector's arc run along the wall on
    # top of the column it is heading for, so that stretch gets a bead twice.
    # Trimming here rather than after assembly keeps the column-to-connector
    # pairing, which is what the trim needs to know.
    trimmed_columns = [column_info["trimmed_geometry"] for column_info in oriented_info]
    trimmed_columns, connector_slots, _ = repair_snake_overlaps(
        trimmed_columns,
        connector_slots,
        infill_boundary,
        margin=line_width + wall_spacing,
    )
    for column_info, trimmed in zip(oriented_info, trimmed_columns):
        column_info["trimmed_geometry"] = trimmed
    connectors = [connector for connector in connector_slots if connector is not None]

    ordered_segments = []
    parts: list[SnakePart] = []
    for index, column_info in enumerate(oriented_info):
        ordered_segments.append(column_info["trimmed_geometry"])
        parts.append(SnakePart("column", column_info["trimmed_geometry"], index))
        slot = connector_slots[index] if index < len(connector_slots) else None
        if slot is not None:
            ordered_segments.append(slot)
            parts.append(SnakePart("connector", slot, index, index + 1))

    flat_segments = []
    for geometry in ordered_segments:
        if geometry.is_empty:
            continue
        if geometry.geom_type == "LineString":
            flat_segments.append(geometry)
        elif geometry.geom_type == "MultiLineString":
            flat_segments.extend(geometry.geoms)
        elif geometry.geom_type == "GeometryCollection":
            for sub_geometry in geometry.geoms:
                if sub_geometry.geom_type == "LineString":
                    flat_segments.append(sub_geometry)
                elif sub_geometry.geom_type == "MultiLineString":
                    flat_segments.extend(sub_geometry.geoms)

    snake_path = linemerge(flat_segments)

    ordered_coords: list[tuple[float, float]] = []

    def _extract_coords(geometry: base.BaseGeometry) -> None:
        if geometry.is_empty:
            return
        if hasattr(geometry, "geoms"):
            for sub_geometry in geometry.geoms:
                _extract_coords(sub_geometry)
        elif hasattr(geometry, "coords"):
            coords = list(geometry.coords)
            if ordered_coords and coords:
                first = coords[0]
                last = ordered_coords[-1]
                if (
                    abs(first[0] - last[0]) < GeometrySettings.LENGTH_EPSILON
                    and abs(first[1] - last[1]) < GeometrySettings.LENGTH_EPSILON
                ):
                    coords = coords[1:]
            ordered_coords.extend(coords)

    for geometry in ordered_segments:
        _extract_coords(geometry)

    clipped_columns_union = unary_union([column["geometry"] for column in oriented_info])
    connectors_union = unary_union(connectors) if connectors else GeometryCollection()
    result = (clipped_columns_union, connectors_union, snake_path, start_point, ordered_coords)
    return (*result, parts) if with_parts else result

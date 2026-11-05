"""
Remove the material a snake connector deposits twice along an oblique wall.

``_connect_points_inside_boundary`` builds a connector as ``spoke_out + arc +
spoke_in``, where the arc is a slice of the infill boundary. On an axis-aligned
edge that arc is a short hop between neighbouring columns and nothing overlaps.
On an oblique or curved edge it runs *along* the wall, on top of the column it
is heading for, and the bead is laid down twice over that stretch. Measured on
the outline library at 7 mm line width: 488 mm of a 494 mm connector sat inside
the column's band on the inner wall of an L.

The three rules, in the order they apply:

1. **Which connectors are at risk** -- every sub-segment of the arc is measured
   against the *global* axes. Straight and axis-aligned is safe; at an angle
   means the arc follows an oblique edge, and a run of small turns means it
   follows a curve. One test covers edges, corners and curves.
2. **Which column it doubles** -- the distance from the connector's middle to
   each of the two columns it joins. The middle excludes one margin at each end,
   so the legitimate junctions do not register.
3. **What comes off** -- the connector is cut where it enters the column's band
   and landed on the column, and the column keeps the span between its outermost
   cuts.

Two things this got wrong before, both worth keeping written down:

* deciding the column's kept piece *per connector* let the connector at one end
  write the whole column back and undo the cut made at the other end, which left
  a 12.53 mm stub dangling;
* measuring sagitta over the whole connector is useless, because every connector
  carries the same little U-step, put there by the spokes and not by the arc.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, Point, base
from shapely.ops import substring

AXIS_TOLERANCE_DEG = 1.0        # within this of 0 or 90 counts as axis-aligned
ON_BOUNDARY_MM = 0.5            # how close an arc vertex sits to the boundary
TOUCH_EPSILON_MM = 1e-6
MAX_CUTS_PER_CONNECTOR = 2  # a connector can double both of its columns


@dataclass(frozen=True)
class RepairStats:
    """What the pass did, for metrics, for tests and for drawing.

    ``removed`` carries the actual pieces taken off. They cannot be recovered by
    comparing the old and new toolpaths: what this pass removes is material that
    was laid *on top of* geometry that stays, so a distance test finds it at zero
    distance from the result. It has to come from the repair itself.
    """

    flagged: int = 0
    repaired: int = 0
    connector_mm: float = 0.0
    column_mm: float = 0.0
    removed: tuple = ()


def _points(geometry) -> list:
    """Every point in a geometry, whatever container shapely wrapped it in."""
    if geometry is None or geometry.is_empty:
        return []
    parts = geometry.geoms if hasattr(geometry, "geoms") else [geometry]
    out = []
    for part in parts:
        if part.geom_type == "Point":
            out.append(part)
        elif part.geom_type == "LineString":
            out.extend(Point(c) for c in (part.coords[0], part.coords[-1]))
    return out


def _lines(geometry) -> list[LineString]:
    if geometry is None or geometry.is_empty:
        return []
    parts = geometry.geoms if hasattr(geometry, "geoms") else [geometry]
    return [part for part in parts if part.geom_type == "LineString"]


def arc_off_axis(connector: LineString, boundary: base.BaseGeometry) -> float:
    """Worst deviation from the global axes among the arc's sub-segments.

    The arc is the connector minus its first and last vertex: those two are the
    spokes that reach out to the columns. Taking it by construction rather than
    by proximity matters -- picking vertices near the boundary catches a spoke
    end that happens to pass close to a *different* wall, which reported 33.69
    deg on a plain rectangle.
    """
    coords = np.asarray(connector.coords)
    if len(coords) < 4:
        return 0.0                                   # no arc of its own
    arc = coords[1:-1]
    if len(arc) < 2:
        return 0.0
    if not all(boundary.distance(Point(point)) <= ON_BOUNDARY_MM for point in arc):
        return 0.0
    worst = 0.0
    for start, end in zip(arc[:-1], arc[1:]):
        delta = end - start
        if np.linalg.norm(delta) < 1e-9:
            continue
        angle = abs(np.degrees(np.arctan2(delta[1], delta[0]))) % 180.0
        worst = max(worst, min(angle, abs(angle - 90.0), abs(angle - 180.0)))
    return worst


def _cut_connector(connector: LineString, anchor: LineString, crossing: LineString,
                   domain: base.BaseGeometry, margin: float, obstacles=()):
    """Cut where the connector enters the crossing column's band, keep the anchor side.

    Returns ``(kept_connector, dropped_tail, parameter_on_column)`` or ``None``.
    The column is deliberately left alone here: see ``_trim_column``.
    """
    coords = list(connector.coords)
    if Point(coords[0]).distance(anchor) > Point(coords[-1]).distance(anchor):
        connector = LineString(coords[::-1])

    outside = _lines(connector.difference(crossing.buffer(margin)))
    if not outside:
        return None
    first = min(outside, key=lambda line: connector.project(Point(line.coords[0])))
    cut_distance = connector.project(Point(first.coords[-1]))
    if cut_distance <= TOUCH_EPSILON_MM:
        return None

    kept = substring(connector, 0.0, cut_distance)

    # Land ON the column rather than stopping at the edge of the band. Cutting at
    # the band boundary leaves a gap of exactly one margin at every break, which
    # is what split the stroke into fragments.
    band_end = Point(kept.coords[-1])
    parameter = crossing.project(band_end)
    cut_point = crossing.interpolate(parameter)
    landed = list(kept.coords)
    if band_end.distance(cut_point) > 1e-9:
        # The landing is a straight chord, and at a re-entrant corner -- the
        # inside of an L -- a chord between two points that are both in the part
        # can still pass outside it. Measured: 6 segments outside the domain
        # across the outline library before this test was added.
        chord = LineString([band_end, cut_point])
        # One margin exactly is the normal case, not an outlier: the cut sits on
        # the edge of the band and the landing is its perpendicular foot on the
        # column, so the chord measures the margin to within float noise. A
        # strict ``>`` here rejected the typical repair and left the defect in
        # place on trapezoid, pentagon and octagon alike.
        if chord.length > margin + TOUCH_EPSILON_MM:
            return None
        # The chord is a new segment up to one bead long, so it can cut straight
        # through a line that happens to lie between the connector and the
        # column. Rejecting here sends the caller to the narrower band, whose
        # chord is half as long and has room to pass.
        if any(chord.crosses(other) for other in obstacles):
            return None
        if not domain.covers(chord):
            # Inside a re-entrant corner the chord can leave the part even though
            # both of its ends are in it. Where the connector already lies on the
            # column -- which is the case that matters, since that is the doubled
            # material -- cut at the contact instead and no chord is needed.
            contact = connector.intersection(crossing)
            points = [Point(p) for p in _points(contact)]
            if not points:
                return None
            first_contact = min(points, key=connector.project)
            cut_distance = connector.project(first_contact)
            if cut_distance <= TOUCH_EPSILON_MM:
                return None
            kept = substring(connector, 0.0, cut_distance)
            parameter = crossing.project(first_contact)
            cut_point = crossing.interpolate(parameter)
            landed = list(kept.coords)
        landed.append((cut_point.x, cut_point.y))
    kept_connector = LineString(landed)
    # A repaired connector has to stay simple. Cutting at the wide band can
    # leave one that doubles back across its own arc -- one such case per 144
    # cells, and it was the last crossing standing. Rejecting sends the caller
    # to the narrower band.
    if not kept_connector.is_simple:
        return None
    # And it must not still cross the column it was doubling. Cutting at the
    # wide band can take only the last couple of millimetres while the arc
    # carries on across the column further back, which leaves the crossing
    # exactly where it was.
    if kept_connector.crosses(crossing):
        return None
    tail = substring(connector, cut_distance, connector.length)
    return kept_connector, tail, parameter


def _trim_column(column: LineString, cuts: list[tuple[float, bool]]):
    """Keep the span between the outermost cuts; return the stubs separately.

    Which side of a cut is spare is fixed by the direction of travel, not by
    which piece is shorter. When the connector doubles back over the column it
    just *left*, the duplicated material runs from the landing to that column's
    end, so the column has to finish at the landing for the path to carry on
    into the connector. When it doubles over the column it is heading *to*, the
    duplicate runs from that column's start to the landing.

    Choosing the shorter piece instead put the landing on the wrong end of the
    column often enough that the junction check threw the whole repair away --
    which is why trapezoid, pentagon and octagon kept their defects while the
    pass reported them as flagged.

    All the cuts on a column are applied together, so the connector at one end
    cannot undo what the connector at the other end removed.
    """
    low, high = 0.0, column.length
    for parameter, doubles_back in cuts:
        if doubles_back:
            high = min(high, parameter)        # the column must end at the landing
        else:
            low = max(low, parameter)          # the column must start at the landing
    if high - low <= TOUCH_EPSILON_MM:
        return column, []
    stubs = []
    if low > TOUCH_EPSILON_MM:
        stubs.append((substring(column, 0.0, low), low))
    if high < column.length - TOUCH_EPSILON_MM:
        stubs.append((substring(column, high, column.length), high))
    return substring(column, low, high), stubs


def _dead_end(stub: LineString, cut: Point, neighbours, margin: float) -> bool:
    """True when nothing but the cut itself attaches to this stub."""
    tail = stub.difference(cut.buffer(margin))
    if tail.is_empty:
        return True
    return all(other.distance(tail) > TOUCH_EPSILON_MM for other in neighbours)


def repair_snake_overlaps(
        columns: list[base.BaseGeometry],
        connectors: list[base.BaseGeometry | None],
        infill_boundary: base.BaseGeometry,
        margin: float,
) -> tuple[list[base.BaseGeometry], list[base.BaseGeometry | None], RepairStats]:
    """Trim the connectors that double a column, and the stubs they leave behind.

    ``connectors[i]`` bridges ``columns[i]`` and ``columns[i + 1]``; a ``None``
    slot means no connector was produced for that gap. Both lists come back the
    same length, so the caller's column-to-connector pairing survives.

    A cut is only kept when it produces a clean junction: the connector has to
    land on an *end* of the trimmed column. Landing part way along one adds a T
    that the path then visits twice, which is a crossing -- 15 cells of the
    outline library gained crossings they did not have before this was enforced.
    """
    if margin <= 0.0 or len(columns) < 2:
        return list(columns), list(connectors), RepairStats()

    # Two different quantities, and using one number for both was a mistake.
    # ``margin`` is the contact band: one bead, because two parallel centrelines
    # that far apart touch edge to edge and any closer they overlap. ``anchor``
    # is how much of each connector end counts as its legitimate junction with a
    # column, and it has to stay at half a bead -- widening it left connectors
    # shorter than two beads with no middle to measure at all.
    anchor_margin = margin / 2.0
    boundary = infill_boundary.boundary
    domain = infill_boundary.buffer(TOUCH_EPSILON_MM)
    kept_connectors = list(connectors)
    cuts: dict[int, list] = {}                     # slot -> [(column, parameter, left)]
    tails: dict[int, list] = {}                     # slot -> pieces of connector dropped
    fallback: dict[int, tuple] = {}                 # slot -> the one-cut result
    flagged = 0

    for slot, connector in enumerate(connectors):
        if connector is None or connector.geom_type != "LineString":
            continue
        if slot + 1 >= len(columns):
            continue
        if arc_off_axis(connector, boundary) <= AXIS_TOLERANCE_DEG:
            continue
        flagged += 1
        left, right = columns[slot], columns[slot + 1]
        if left.geom_type != "LineString" or right.geom_type != "LineString":
            continue
        # Distance from the connector's MIDDLE to each of its own two columns.
        # Counting the pieces left after subtracting the band only catches a
        # connector that leaves the band and comes back; one that enters and
        # stays inside yields a single piece and slips through -- 26 of 38 missed
        # on an ellipse before this was measured directly.
        middle = connector
        if middle.length > 2.0 * anchor_margin:
            middle = substring(middle, anchor_margin, middle.length - anchor_margin)
        left_distance = middle.distance(left)
        right_distance = middle.distance(right)
        if min(left_distance, right_distance) >= margin:  # contact band
            continue
        obstacles = [line for position, line in enumerate(columns)
                     if position not in (slot, slot + 1)]
        obstacles += [line for position, line in enumerate(connectors)
                      if line is not None and position != slot]

        # A connector can double BOTH of its columns, and treating only the
        # nearer one left the other overlap in place. Cut each doubled end
        # independently: the path then reads column i, the middle of the
        # connector, column i+1, with both columns finishing on their landing.
        # The right end is cut first so the piece keeps its original direction;
        # the left cut then works on what is left of it.
        working = connector
        entries: list[tuple[int, float, bool]] = []
        # Nearest column first. Cutting against a column the connector merely
        # grazes can consume the piece before the real overlap is reached, which
        # cost 12 m of duplicated bead when the order was fixed rather than
        # sorted.
        sides = sorted(((False, slot + 1, right_distance),
                        (True, slot, left_distance)), key=lambda side: side[2])
        for doubles_back, index, distance in sides[:MAX_CUTS_PER_CONNECTOR]:
            if distance >= margin:
                continue
            crossing = columns[index]
            keep_side = columns[slot] if doubles_back is False else columns[slot + 1]
            # Cut where the overlap STARTS, which is the edge of the contact
            # band -- one bead, since that is where two parallel beads begin to
            # share material. Trying half a bead first instead left everything
            # between the two bands in place and doubled the residual overlaps,
            # 92 against 47. Half a bead is only the fallback, for a connector
            # that lies entirely inside the wide band and leaves nothing to cut
            # there.
            outcome = _cut_connector(working, keep_side, crossing, domain,
                                     margin, obstacles)
            if outcome is None:
                outcome = _cut_connector(working, keep_side, crossing, domain,
                                         anchor_margin, obstacles)
            if outcome is None:
                continue
            piece, tail, parameter = outcome
            if doubles_back:
                piece = LineString(list(piece.coords)[::-1])
            tails.setdefault(slot, []).append(tail)
            working = piece
            entries.append((index, parameter, doubles_back))
            if len(entries) == 1:
                # Keep the one-cut result. If cutting the second end makes the
                # pair fail validation, the slot degrades to this instead of
                # losing the repair altogether -- discarding both cost 12 m of
                # duplicated bead and tripled the residual overlaps at d55.
                fallback[slot] = (working, list(entries), list(tails[slot]))

        if not entries:
            continue
        kept_connectors[slot] = working
        cuts[slot] = entries

    # Trim, check every junction landed on the column end the path continues
    # from, drop the cuts that did not, trim again. Dropping one cut changes
    # where its column is trimmed, which can unseat another junction, so this
    # repeats until it settles.
    repaired_columns, stubs = list(columns), []
    while cuts:
        repaired_columns, stubs = _apply_cuts(columns, cuts, kept_connectors, margin)
        rejected = [slot for slot, entries in cuts.items()
                    if not all(_joins_cleanly(kept_connectors[slot],
                                              repaired_columns[index], doubles_back,
                                              anchor_margin)
                               for index, _, doubles_back in entries)]
        if not rejected:
            break
        for slot in rejected:
            if len(cuts[slot]) > 1 and slot in fallback:
                kept_connectors[slot], cuts[slot], tails[slot] = fallback.pop(slot)
                continue
            kept_connectors[slot] = connectors[slot]
            del cuts[slot]
            tails.pop(slot, None)
    else:
        repaired_columns, stubs = list(columns), []

    # Snap each landing onto the column's own coordinate. project() and
    # interpolate() round-trip through parameter space and land 2.8e-14 mm off --
    # invisible, but the assembler needs exact equality to chain the pieces, and
    # reported the stroke as two fragments that were already touching.
    for slot, entries in cuts.items():
        landed = list(kept_connectors[slot].coords)
        for index, _, doubles_back in entries:
            column = repaired_columns[index]
            if doubles_back:
                landed[0] = tuple(column.coords[-1])   # column ends, connector starts
            else:
                landed[-1] = tuple(column.coords[0])   # connector ends, column starts
        kept_connectors[slot] = LineString(landed)

    return repaired_columns, kept_connectors, RepairStats(
        flagged=flagged, repaired=len(cuts),
        connector_mm=sum(connectors[slot].length - kept_connectors[slot].length
                         for slot in cuts),
        column_mm=sum(columns[index].length - repaired_columns[index].length
                      for index in {index for entries in cuts.values()
                                    for index, _, _ in entries}),
        removed=tuple(t for pieces in tails.values() for t in pieces) + tuple(stubs))


def _joins_cleanly(connector, column, left_doubled: bool, margin: float) -> bool:
    """Does the junction land on the column end the path actually continues from?

    Either end is not good enough. Landing on the far end makes the path jump the
    whole length of the column to reach the end it is assembled from, and landing
    part way along adds a T that the path visits twice.
    """
    if connector is None or column.geom_type != "LineString":
        return False
    if left_doubled:
        junction, target = connector.coords[0], column.coords[-1]
    else:
        junction, target = connector.coords[-1], column.coords[0]
    return float(np.linalg.norm(np.asarray(junction) - np.asarray(target))) <= margin


def _apply_cuts(columns, cuts, kept_connectors, margin: float):
    """Trim every column that carries cuts, dropping only the stubs nothing hangs off.

    Returns the new column list and the stubs actually removed.
    """
    pending: dict[int, list[tuple[float, bool]]] = {}
    for entries in cuts.values():
        for index, parameter, doubles_back in entries:
            pending.setdefault(index, []).append((parameter, doubles_back))

    surviving = [line for line in kept_connectors if line is not None]
    repaired, dropped = list(columns), []
    for index, column_cuts in pending.items():
        original = columns[index]
        kept_column, stubs = _trim_column(original, column_cuts)
        neighbours = [line for position, line in enumerate(columns) if position != index]
        neighbours += surviving
        spare = [(stub, cut) for stub, cut in stubs
                 if _dead_end(stub, original.interpolate(cut), neighbours, margin)]
        if len(spare) < len(stubs):
            keep = {round(cut, 9) for _, cut in spare}
            kept_column, _ = _trim_column(
                original, [c for c in column_cuts if round(c[0], 9) in keep])
        if kept_column.length > TOUCH_EPSILON_MM:     # never delete a whole column
            repaired[index] = kept_column
            dropped.extend(stub for stub, _ in spare)
    return repaired, dropped

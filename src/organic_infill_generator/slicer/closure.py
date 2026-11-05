"""
Contour closure: from the raw segments of one layer to closed polygons.

A plane cut through a mesh returns unordered segments. ``slice_contours`` hands
them to a :class:`ContourCloser`, so the way a layer is closed is a replaceable
step behind one interface.

:class:`SimpleContourCloser` is the implementation shipped here. It joins the
segments whose endpoints coincide into cycles and sets aside whatever cannot
close. It never invents geometry: a layer with a real gap in the mesh keeps its
gap, reported as open chains, rather than a guessed bridge across it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from shapely.geometry import Polygon

from .config import ENDPOINT_TOLERANCE


@dataclass
class ClosureResult:
    """Outcome of closing one layer."""

    polygons: list[Polygon] = field(default_factory=list)  # one polygon per closed cycle
    open_chains: list[np.ndarray] = field(default_factory=list)  # (N, 2) chains that do not close

    @property
    def has_open_ends(self) -> bool:
        return bool(self.open_chains)


@runtime_checkable
class ContourCloser(Protocol):
    """Turns the raw segments of one layer into closed polygons."""

    def close(self, segments: np.ndarray, model_diag: float) -> ClosureResult:
        """
        Close ``segments``, an ``(N, 2, 2)`` array of XY segments.

        ``model_diag`` is the diagonal of the model, for implementations whose
        tolerances scale with the part.
        """
        ...


def _weld_endpoints(points: np.ndarray, tolerance: float) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """
    Map every point to a node, merging the points within ``tolerance``.

    The two faces sharing a mesh edge compute the same intersection up to float
    noise. Rounding to a grid is not enough to merge them: two points 1e-12
    apart can still fall on either side of a cell boundary, which would leave a
    false open end. So the neighbouring cells are searched as well.

    The first point seen of each node keeps its exact coordinates, which makes
    the result independent of the tolerance grid.
    """
    labels = np.empty(len(points), dtype=np.int64)
    nodes: list[tuple[float, float]] = []
    cells: dict[tuple[int, int], list[int]] = {}

    for index, (x, y) in enumerate(points.tolist()):
        cell_x = math.floor(x / tolerance)
        cell_y = math.floor(y / tolerance)
        match = None
        for neighbour_x in (cell_x - 1, cell_x, cell_x + 1):
            for neighbour_y in (cell_y - 1, cell_y, cell_y + 1):
                for node in cells.get((neighbour_x, neighbour_y), ()):
                    node_x, node_y = nodes[node]
                    if math.hypot(x - node_x, y - node_y) <= tolerance:
                        match = node
                        break
                if match is not None:
                    break
            if match is not None:
                break
        if match is None:
            match = len(nodes)
            nodes.append((x, y))
            cells.setdefault((cell_x, cell_y), []).append(match)
        labels[index] = match
    return labels, nodes


def _build_adjacency(labels: np.ndarray) -> dict[int, set[int]]:
    """Undirected graph of the segments, without collapsed or repeated edges."""
    adjacency: dict[int, set[int]] = {}
    for start, end in labels.reshape(-1, 2).tolist():
        if start == end:
            continue
        adjacency.setdefault(start, set()).add(end)
        adjacency.setdefault(end, set()).add(start)
    return adjacency


def _remove_edge(adjacency: dict[int, set[int]], first: int, second: int) -> None:
    adjacency[first].discard(second)
    adjacency[second].discard(first)


def _peel_open_chains(adjacency: dict[int, set[int]]) -> list[list[int]]:
    """
    Remove every dangling chain from the graph and return it.

    A chain starts at a node with one edge. Taking the chains out first means
    the cycle walk below never wanders into a dead end and loses the cycle the
    chain was attached to.
    """
    chains: list[list[int]] = []
    for start in sorted(adjacency):
        while len(adjacency[start]) == 1:
            chain = [start]
            current = start
            while len(adjacency[current]) == 1:
                following = min(adjacency[current])
                _remove_edge(adjacency, current, following)
                chain.append(following)
                current = following
            chains.append(chain)
    return chains


def _extract_cycles(adjacency: dict[int, set[int]]) -> tuple[list[list[int]], list[list[int]]]:
    """
    Decompose what is left of the graph into simple cycles.

    The walk follows unused edges and cuts a cycle off as soon as it reaches a
    node already on its path, so two contours touching at one point come out as
    two rings. Lowest node first, which keeps the output deterministic.
    """
    cycles: list[list[int]] = []
    dead_ends: list[list[int]] = []

    for start in sorted(adjacency):
        while adjacency[start]:
            path = [start]
            positions = {start: 0}
            while True:
                current = path[-1]
                if not adjacency[current]:
                    if len(path) > 1:
                        dead_ends.append(path)
                    break
                following = min(adjacency[current])
                _remove_edge(adjacency, current, following)
                if following in positions:
                    cut = positions[following]
                    cycles.append(path[cut:])
                    for node in path[cut + 1:]:
                        del positions[node]
                    del path[cut + 1:]
                    if len(path) == 1 and not adjacency[start]:
                        break
                else:
                    positions[following] = len(path)
                    path.append(following)
    return cycles, dead_ends


@dataclass(frozen=True)
class SimpleContourCloser:
    """
    Join segments into cycles and drop the open chains.

    ``tolerance`` is the distance under which two endpoints count as the same
    point, in model units.
    """

    tolerance: float = ENDPOINT_TOLERANCE

    def close(self, segments: np.ndarray, model_diag: float = 0.0) -> ClosureResult:
        if self.tolerance <= 0.0:
            raise ValueError("'tolerance' must be > 0.")
        segments = np.asarray(segments, dtype=float)
        if segments.size == 0:
            return ClosureResult()

        labels, nodes = _weld_endpoints(segments.reshape(-1, 2), self.tolerance)
        adjacency = _build_adjacency(labels)
        chains = _peel_open_chains(adjacency)
        cycles, dead_ends = _extract_cycles(adjacency)

        return ClosureResult(
            polygons=[
                Polygon([nodes[node] for node in cycle])
                for cycle in cycles
                if len(cycle) >= 3
            ],
            open_chains=[
                np.asarray([nodes[node] for node in chain], dtype=float)
                for chain in [*chains, *dead_ends]
            ],
        )

"""
The global pattern plan keeps the cells of every layer stacked.

Measured as in the public deliverable: a drafted part whose contour shrinks by a
different amount on each side, sliced into 12 layers. With the global plan every
layer is a clip of one field, so a cell wall of the top layer lies on the same
wall of the bottom layer. A raster computed in each layer's own bounding box
drifts instead.
"""
from __future__ import annotations

import unittest

import numpy as np
from _fixtures import layer, request, slicer_output
from shapely.geometry import LineString, Point, box

from organic_infill_generator.infill_generator.generator import (
    apply_global_pattern_to_layer,
    build_global_pattern_plan,
)
from organic_infill_generator.infill_generator.patterns.pattern_base import DensityProfile
from organic_infill_generator.infill_generator.patterns.pattern_builder import resolve_pattern_request
from organic_infill_generator.infill_generator.patterns.sequence_drawer import build_infill_columns


LAYERS = 12


def drafted_layers() -> list[dict]:
    """A 12-layer stack: 150 x 100 mm at the bottom, a different draft per side."""
    stack = []
    for index in range(LAYERS):
        t = index / (LAYERS - 1)
        left, right = 40.0 * t, 150.0 - 18.0 * t
        low, high = 10.0 * t, 100.0 - 10.0 * t
        stack.append(layer(0.75 + 1.5 * index, [[left, low], [right, low], [right, high], [left, high]]))
    return stack


def local_columns(resolved, bounds) -> list[LineString]:
    """The conventional raster: columns computed in one layer's own bounding box."""
    return build_infill_columns(
        infill_boundary=box(*bounds),
        pattern_sequence=resolved.pattern_sequence,
        infill_spacing=resolved.infill_spacing,
        pattern_amplitude=resolved.pattern_amplitude,
        pattern_step=resolved.pattern_step,
        density_profile=DensityProfile(global_density_percent=resolved.estimated_density),
        line_width=resolved.line_width,
        column_clearance=resolved.wall_spacing,
    )


class TestGlobalPlanAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.layers = drafted_layers()
        cls.output = slicer_output(cls.layers)
        cls.plan = build_global_pattern_plan(
            request(num_walls=0, fillet_radius=0.0, wall_spacing=0.0, extrude_height=18.0),
            cls.output,
        )

    def test_every_layer_infill_lies_on_the_global_field(self):
        for item in self.layers:
            preview = apply_global_pattern_to_layer(self.plan, item)
            columns = self._column_midpoints(preview)
            self.assertGreater(len(columns), 2)
            for point in columns:
                self.assertLess(self._distance_to_field(point), 1e-6)

    def test_all_layers_share_the_column_positions(self):
        bottom = self._occupied_columns(self.layers[0])
        top = self._occupied_columns(self.layers[-1])
        self.assertTrue(top <= bottom)
        self.assertGreaterEqual(len(top), 2)

    def test_measured_drift_is_zero_with_the_plan(self):
        drift = max(self._field_offset(self.plan.raw_columns, item) for item in self.layers)
        self.assertLess(drift, 1e-6)

    def test_a_raster_per_layer_drifts(self):
        """The control: without this the zero above would prove nothing."""
        resolved = resolve_pattern_request(request(num_walls=0, fillet_radius=0.0, wall_spacing=0.0))
        top = np.asarray(self.layers[-1]["contours"][0]["exterior"])
        local = local_columns(resolved, (*top.min(axis=0), *top.max(axis=0)))

        drift = self._field_offset(local, self.layers[-1])

        self.assertGreater(drift, resolved.line_width / 2.0)

    def test_plan_is_built_once_and_does_not_depend_on_the_layer_order(self):
        shuffled = slicer_output(list(reversed(self.layers)))
        other = build_global_pattern_plan(
            request(num_walls=0, fillet_radius=0.0, wall_spacing=0.0, extrude_height=18.0),
            shuffled,
        )
        self.assertEqual(len(other.raw_columns), len(self.plan.raw_columns))
        for first, second in zip(other.raw_columns, self.plan.raw_columns):
            self.assertTrue(first.equals_exact(second, 0.0))

    def _column_midpoints(self, preview) -> list[Point]:
        """Sample the infill away from the connectors, which leave the field by design."""
        contour = box(*preview.infill_boundary.bounds).buffer(-preview.line_width * 2.0)
        path = np.asarray(preview.infill_path_points)
        middles = (path[:-1] + path[1:]) / 2.0
        return [Point(xy) for xy in middles if contour.contains(Point(xy))]

    def _distance_to_field(self, point: Point, columns=None) -> float:
        columns = self.plan.raw_columns if columns is None else columns
        return min(column.distance(point) for column in columns)

    def _occupied_columns(self, item: dict) -> set[int]:
        preview = apply_global_pattern_to_layer(self.plan, item)
        occupied = set()
        for point in self._column_midpoints(preview):
            nearest = min(
                range(len(self.plan.raw_columns)),
                key=lambda index: self.plan.raw_columns[index].distance(point),
            )
            occupied.add(nearest)
        return occupied

    def _field_offset(self, columns: list[LineString], item: dict) -> float:
        """Largest distance from the printed cell walls to the columns of the global field."""
        preview_layer = apply_global_pattern_to_layer(self.plan, item)
        sample = self._sample_columns(columns, preview_layer.infill_boundary)
        return max(self._distance_to_field(point) for point in sample)

    @staticmethod
    def _sample_columns(columns: list[LineString], boundary) -> list[Point]:
        inner = box(*boundary.bounds).buffer(-1.0)
        points = []
        for column in columns:
            coords = np.asarray(column.coords)
            for xy in (coords[:-1] + coords[1:]) / 2.0:
                point = Point(xy)
                if inner.contains(point):
                    points.append(point)
        return points


if __name__ == "__main__":
    unittest.main()

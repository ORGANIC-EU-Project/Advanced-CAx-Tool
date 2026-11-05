"""Every layer must print as one continuous path: no jump, no leaving the part."""
from __future__ import annotations

import logging
import unittest

import numpy as np
from _fixtures import layer, request, slicer_output
from shapely.geometry import LineString, Polygon

from organic_infill_generator.infill_generator.generator import (
    _build_layer_outputs,
    apply_global_pattern_to_layer,
    build_global_pattern_plan,
    generate_pattern,
)
from organic_infill_generator.infill_generator.settings import PatternCatalog
from organic_infill_generator.infill_generator.validation import (
    coverage_metrics,
    eulerian_metrics,
    ordered_path_continuity_metrics,
    path_quality_metrics,
)

SHAPES = {
    "square": [[0.0, 0.0], [240.0, 0.0], [240.0, 240.0], [0.0, 240.0]],
    "octagon": [
        [70.0, 0.0], [170.0, 0.0], [240.0, 70.0], [240.0, 170.0],
        [170.0, 240.0], [70.0, 240.0], [0.0, 170.0], [0.0, 70.0],
    ],
    "ell": [[0.0, 0.0], [240.0, 0.0], [240.0, 100.0], [100.0, 100.0], [100.0, 240.0], [0.0, 240.0]],
}
PATTERNS = sorted(PatternCatalog.PUBLIC_PATTERN_MAP)


def preview_of(shape: str, **overrides):
    output = slicer_output([layer(0.75, SHAPES[shape])])
    plan = build_global_pattern_plan(request(**overrides), output)
    return apply_global_pattern_to_layer(plan, output["layers"][0])


class TestContinuity(unittest.TestCase):
    def test_every_pattern_on_every_shape_has_no_jump(self):
        for shape in SHAPES:
            for pattern in PATTERNS:
                for walls in (0, 1, 2):
                    with self.subTest(shape=shape, pattern=pattern, walls=walls):
                        preview = preview_of(shape, infill_pattern=pattern, num_walls=walls)
                        metrics = ordered_path_continuity_metrics(preview.path_points, tolerance=1e-6)
                        self.assertEqual(metrics["jump_count"], 0)
                        self.assertTrue(metrics["is_single_path"])
                        self.assertGreater(len(preview.path_points), 2)

    def test_path_is_one_polyline_with_no_repeated_points(self):
        preview = preview_of("square", infill_pattern="honeycomb")
        steps = np.linalg.norm(np.diff(preview.path_points, axis=0), axis=1)
        self.assertGreater(steps.min(), 0.0)

    def test_centerline_stays_inside_the_contour(self):
        for shape, points in SHAPES.items():
            contour = Polygon(points).buffer(1e-6)
            for pattern in PATTERNS:
                with self.subTest(shape=shape, pattern=pattern):
                    preview = preview_of(shape, infill_pattern=pattern, num_walls=1)
                    self.assertTrue(contour.covers(LineString(preview.path_points)))

    def test_infill_keeps_half_a_bead_from_the_outline(self):
        preview = preview_of("square", infill_pattern="diamond", num_walls=0)
        distance = Polygon(SHAPES["square"]).exterior.distance(LineString(preview.path_points))
        self.assertGreaterEqual(distance, 3.0 - 1e-6)

    def test_walls_are_entered_from_the_infill_without_a_jump(self):
        preview = preview_of("square", num_walls=2)
        self.assertEqual(len(preview.wall_centerlines), 2)
        self.assertGreater(len(preview.fillet_protected_indices), 0)
        self.assertEqual(
            ordered_path_continuity_metrics(preview.path_points, tolerance=1e-6)["jump_count"], 0,
        )

    def test_rounded_path_is_as_continuous_as_the_sharp_one(self):
        preview = preview_of("ell", fillet_radius=6.0)
        rounded = np.asarray(preview.mesh_preview_geometry.coords)
        self.assertEqual(ordered_path_continuity_metrics(rounded, tolerance=1e-6)["jump_count"], 0)
        turns = np.diff(rounded, axis=0)
        self.assertGreater(len(rounded), len(preview.path_points))
        self.assertGreater(np.linalg.norm(turns, axis=1).min(), 0.0)

    def test_every_layer_of_a_stack_is_continuous(self):
        layers = [layer(0.75 + 1.5 * index, [
            [2.0 * index, 0.0], [240.0 - index, 0.0], [240.0 - index, 240.0], [2.0 * index, 240.0],
        ]) for index in range(6)]
        output = slicer_output(layers)
        plan = build_global_pattern_plan(request(extrude_height=9.0), output)

        previews = _build_layer_outputs(plan, output, title="stack")

        self.assertEqual(len(previews), 6)
        for preview in previews:
            self.assertEqual(
                ordered_path_continuity_metrics(preview.path_points, tolerance=1e-6)["jump_count"], 0,
            )


class TestOverlapRepair(unittest.TestCase):
    def test_oblique_walls_are_not_printed_twice(self):
        """Connectors running along an oblique edge used to lay the bead twice."""
        preview = preview_of("octagon", infill_pattern="vertical", num_walls=0, infill_density=30.0)

        quality = path_quality_metrics(
            path_points=preview.infill_path_points,
            printable_geometry=preview.infill_boundary,
            line_width=preview.line_width,
        )

        self.assertEqual(quality["self_overlap_count"], 0)
        self.assertEqual(quality["crossing_count"], 0)


class TestValidationMetrics(unittest.TestCase):
    def test_a_jump_is_counted(self):
        broken = [np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([[5.0, 0.0], [6.0, 0.0]])]
        metrics = ordered_path_continuity_metrics(broken, tolerance=1e-6)
        self.assertEqual(metrics["jump_count"], 1)
        self.assertFalse(metrics["is_single_path"])

    def test_eulerian_metrics_of_a_simple_path(self):
        metrics = eulerian_metrics(np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]))
        self.assertEqual(metrics["graph_nodes"], 3)
        self.assertEqual(metrics["graph_edges"], 2)
        self.assertEqual(metrics["graph_components"], 1)
        self.assertEqual(metrics["odd_degree_nodes"], 2)
        self.assertTrue(metrics["is_eulerian_path_candidate"])

    def test_eulerian_metrics_node_the_crossing(self):
        crossing = [np.array([[0.0, 0.0], [10.0, 10.0]]), np.array([[0.0, 10.0], [10.0, 0.0]])]
        metrics = eulerian_metrics(crossing)
        self.assertEqual(metrics["graph_nodes"], 5)
        self.assertEqual(metrics["graph_edges"], 4)
        self.assertEqual(metrics["odd_degree_nodes"], 4)
        self.assertFalse(metrics["is_eulerian_path_candidate"])

    def test_eulerian_metrics_count_components(self):
        apart = [np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([[5.0, 0.0], [6.0, 0.0]])]
        metrics = eulerian_metrics(apart)
        self.assertEqual(metrics["graph_components"], 2)
        self.assertFalse(metrics["is_eulerian_path_candidate"])

    def test_denser_infill_covers_more_of_the_part(self):
        covered = []
        for density in (20.0, 50.0, 80.0):
            preview = preview_of("square", infill_density=density, infill_pattern="diamond", num_walls=0)
            covered.append(coverage_metrics(
                path_points=preview.infill_path_points,
                line_width=preview.line_width,
                printable_geometry=preview.infill_boundary,
            )["covered_density_percent"])
        self.assertEqual(covered, sorted(covered))
        self.assertGreater(covered[-1], covered[0])


class TestInfillIslands(unittest.TestCase):
    def test_a_contour_split_by_the_offset_fills_the_largest_island(self):
        # Two lobes joined by a neck narrower than the offset.
        dumbbell = [
            [0.0, 0.0], [120.0, 0.0], [120.0, 58.0], [160.0, 58.0], [160.0, 0.0],
            [220.0, 0.0], [220.0, 120.0], [160.0, 120.0], [160.0, 62.0], [120.0, 62.0],
            [120.0, 120.0], [0.0, 120.0],
        ]
        output = slicer_output([layer(0.75, dumbbell)])
        plan = build_global_pattern_plan(request(num_walls=1), output)

        with self.assertLogs("organic_infill_generator.infill_generator.generator", level=logging.WARNING):
            preview = apply_global_pattern_to_layer(plan, output["layers"][0])

        self.assertEqual(
            ordered_path_continuity_metrics(preview.infill_path_points, tolerance=1e-6)["jump_count"], 0,
        )
        self.assertLess(preview.infill_path_points[:, 0].max(), 120.0)

    def test_generation_does_not_fail_on_such_a_part(self):
        dumbbell = [
            [0.0, 0.0], [120.0, 0.0], [120.0, 58.0], [160.0, 58.0], [160.0, 0.0],
            [220.0, 0.0], [220.0, 120.0], [160.0, 120.0], [160.0, 62.0], [120.0, 62.0],
            [120.0, 120.0], [0.0, 120.0],
        ]
        with self.assertLogs("organic_infill_generator.infill_generator.generator", level=logging.WARNING):
            result = generate_pattern(request(num_walls=1), slicer_output([layer(0.75, dumbbell)]), {})
        self.assertEqual(len(result["layers"]), 1)


if __name__ == "__main__":
    unittest.main()

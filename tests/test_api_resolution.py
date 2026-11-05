import os
import importlib.util
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import organic_infill_generator.infill_generator.generator as main_module
import numpy as np
from shapely.geometry import LineString, Point, Polygon, box


from organic_infill_generator.infill_generator.geometry.clipper_module import assemble_eulerian_path, clip_and_reconnect_columns
from organic_infill_generator.common.geometry import polygon_from_points
from organic_infill_generator.infill_generator.models import PatternRequest
from organic_infill_generator.infill_generator.generator import (
    _PatternPreviewResult,
    _build_pattern_preview,
    _build_preview_mesh,
    _build_multilayer_mesh,
    _mesh_export_line_width,
    apply_global_pattern_to_layer,
    build_global_pattern_plan,
    generate_pattern,
)
from organic_infill_generator.infill_generator.geometry.offset_module import generate_wall_centerlines
from organic_infill_generator.infill_generator.patterns.pattern_base import DensityProfile
from organic_infill_generator.infill_generator.patterns.pattern_builder import _build_diagonal_wall_connector, _build_full_toolpath, resolve_pattern_request
from organic_infill_generator.infill_generator.visualization.plot_utils import plot_pattern_result
from organic_infill_generator.infill_generator.patterns.sequence_drawer import build_infill_columns, iter_column_layout
from organic_infill_generator.infill_generator.settings import (
    DensitySettings,
    MeshSettings,
    PatternCatalog,
    PathSettings,
    PlotSettings,
    SinusoidalSettings,
    WallSettings,
)


HAS_CAD_BACKEND = importlib.util.find_spec("cadquery") is not None


class TestPatternApi(unittest.TestCase):
    @staticmethod
    def _slicer_output_with_layers(layers):
        return {
            "layer_height": 0.2,
            "model_bounds": {
                "min": [0.0, 0.0, 0.0],
                "max": [10.0, 8.0, 1.0],
                "size": [10.0, 8.0, 1.0],
            },
            "layers": layers,
            "total_layers": len(layers),
            "closed_layers": len(layers),
            "failed_layers": 0,
            "noise_layers": 0,
        }

    @staticmethod
    def _layer(z, exterior, *, contour_id=0, is_largest=True):
        return {
            "z": z,
            "method": "exact",
            "contours": [
                {
                    "id": contour_id,
                    "area": float(abs(polygon_from_points(np.asarray(exterior, dtype=float)).area)),
                    "is_largest": is_largest,
                    "exterior": exterior,
                    "holes": [],
                }
            ],
            "open_lines": [],
        }

    @staticmethod
    def _build_infill_boundary(request, contour_points):
        contour = polygon_from_points(contour_points)
        _, infill_boundary = generate_wall_centerlines(
            contour,
            num_walls=request.num_walls,
            line_width=request.line_width,
            wall_spacing=request.wall_spacing,
        )
        return infill_boundary

    def test_outer_wall_uses_same_connection_gap_as_inner_walls(self):
        outer = box(0.0, 0.0, 10.0, 10.0)
        inner = box(1.0, 1.0, 9.0, 9.0)

        full_path, _, _, _, _ = _build_full_toolpath(
            infill_path_points=[],
            infill_path_geometry=LineString(),
            wall_geometries=[outer, inner],
            wall_gap_length=0.5,
            wall_connection_angle_degrees=45.0,
        )

        self.assertGreater(len(full_path), 0)
        self.assertNotEqual(tuple(full_path[0]), tuple(full_path[-1]))

    def test_wall_connector_uses_diagonal_entry_when_angle_is_positive(self):
        outer = LineString(list(box(0.0, 0.0, 10.0, 10.0).boundary.coords))
        connector, _ = _build_diagonal_wall_connector(
            current_point=(1.0, 1.0),
            wall_line=outer,
            connection_angle_degrees=45.0,
        )

        self.assertEqual(len(connector), 2)
        dx = abs(connector[1][0] - connector[0][0])
        dy = abs(connector[1][1] - connector[0][1])
        self.assertGreater(dx, 1e-6)
        self.assertGreater(dy, 1e-6)

    def test_infill_start_extends_vertically_toward_innermost_wall(self):
        """Opt-in behaviour: the lead-in is off by default, see the test below."""
        outer = box(0.0, 0.0, 10.0, 10.0)
        inner = box(1.0, 1.0, 9.0, 9.0)
        infill_points = [(5.0, 5.0), (5.0, 6.0)]

        with patch.object(WallSettings, "EXTEND_INFILL_START_TO_WALL", True):
            full_path, _, start_point, _, _ = _build_full_toolpath(
                infill_path_points=infill_points,
                infill_path_geometry=LineString(infill_points),
                wall_geometries=[outer, inner],
                wall_gap_length=0.5,
                wall_connection_angle_degrees=45.0,
            )

        self.assertIsNotNone(start_point)
        self.assertTrue(np.allclose(full_path[0], np.asarray((5.0, 1.5), dtype=float)))
        self.assertTrue(np.allclose(full_path[1], np.asarray(infill_points[0], dtype=float)))
        self.assertTrue(np.allclose(full_path[2], np.asarray(infill_points[1], dtype=float)))

    def test_infill_start_is_not_extended_by_default(self):
        """The probe is vertical, so the gap it leaves shrinks with cos(theta).

        On a wall at 45 degrees a nominal 3.70 mm clearance became 2.62 mm and on
        an ellipse 1.39 mm, which is the bead overlapping the wall. The lead-in
        is not needed -- the infill reaches the wall through the ordinary
        connector -- so it is off, and the path now starts on the infill itself.
        """
        outer = box(0.0, 0.0, 10.0, 10.0)
        inner = box(1.0, 1.0, 9.0, 9.0)
        infill_points = [(5.0, 5.0), (5.0, 6.0)]

        full_path, _, _, _, _ = _build_full_toolpath(
            infill_path_points=infill_points,
            infill_path_geometry=LineString(infill_points),
            wall_geometries=[outer, inner],
            wall_gap_length=0.5,
            wall_connection_angle_degrees=45.0,
        )

        self.assertTrue(np.allclose(full_path[0], np.asarray(infill_points[0], dtype=float)))
        self.assertTrue(np.allclose(full_path[1], np.asarray(infill_points[1], dtype=float)))

    def test_infill_end_and_wall_entry_merge_when_collinear(self):
        wall = box(0.0, 0.0, 10.0, 10.0)
        infill_points = [(4.0, 2.0), (5.0, 2.0), (5.0, 1.0)]

        full_path, _, _, _, _ = _build_full_toolpath(
            infill_path_points=infill_points,
            infill_path_geometry=LineString(infill_points),
            wall_geometries=[wall],
            wall_gap_length=0.5,
            wall_connection_angle_degrees=0.0,
        )

        self.assertTrue(np.allclose(full_path[0], np.asarray((4.0, 2.0), dtype=float)))
        self.assertTrue(np.allclose(full_path[1], np.asarray((5.0, 2.0), dtype=float)))
        self.assertTrue(np.allclose(full_path[2], np.asarray((5.0, 0.0), dtype=float)))
        self.assertFalse(any(np.allclose(point, np.asarray((5.0, 1.0), dtype=float)) for point in full_path))

    def test_wall_gap_is_left_on_opposite_side_of_wall_entry(self):
        wall = box(0.0, 0.0, 10.0, 10.0)
        infill_points = [(4.0, 2.0), (5.0, 2.0), (5.0, 1.0)]

        with patch.object(WallSettings, "SWAP_GAP_SIDE_WITH_INFILL_DIRECTION", True):
            full_path, _, _, _, _ = _build_full_toolpath(
                infill_path_points=infill_points,
                infill_path_geometry=LineString(infill_points),
                wall_geometries=[wall],
                wall_gap_length=0.5,
                wall_connection_angle_degrees=0.0,
            )

        self.assertTrue(np.allclose(full_path[2], np.asarray((5.0, 0.0), dtype=float)))
        self.assertTrue(np.allclose(full_path[3], np.asarray((0.0, 0.0), dtype=float)))
        self.assertTrue(np.allclose(full_path[-1], np.asarray((5.5, 0.0), dtype=float)))

    def test_wall_gap_can_keep_legacy_side_when_swap_is_disabled(self):
        wall = box(0.0, 0.0, 10.0, 10.0)
        infill_points = [(4.0, 2.0), (5.0, 2.0), (5.0, 1.0)]

        with patch.object(WallSettings, "SWAP_GAP_SIDE_WITH_INFILL_DIRECTION", False):
            full_path, _, _, _, _ = _build_full_toolpath(
                infill_path_points=infill_points,
                infill_path_geometry=LineString(infill_points),
                wall_geometries=[wall],
                wall_gap_length=0.5,
                wall_connection_angle_degrees=0.0,
            )

        self.assertTrue(np.allclose(full_path[2], np.asarray((5.0, 0.0), dtype=float)))
        self.assertTrue(np.allclose(full_path[3], np.asarray((10.0, 0.0), dtype=float)))
        self.assertTrue(np.allclose(full_path[-1], np.asarray((4.5, 0.0), dtype=float)))

    def test_positive_wall_connection_angle_flips_side_with_gap_swap(self):
        wall = box(0.0, 0.0, 10.0, 10.0)
        infill_points = [(5.0, 5.0), (5.0, 6.0)]

        with patch.object(WallSettings, "SWAP_GAP_SIDE_WITH_INFILL_DIRECTION", False):
            legacy_path, _, _, _, _ = _build_full_toolpath(
                infill_path_points=infill_points,
                infill_path_geometry=LineString(infill_points),
                wall_geometries=[wall],
                wall_gap_length=0.5,
                wall_connection_angle_degrees=45.0,
            )

        with patch.object(WallSettings, "SWAP_GAP_SIDE_WITH_INFILL_DIRECTION", True):
            swapped_path, _, _, _, _ = _build_full_toolpath(
                infill_path_points=infill_points,
                infill_path_geometry=LineString(infill_points),
                wall_geometries=[wall],
                wall_gap_length=0.5,
                wall_connection_angle_degrees=45.0,
            )

        # Index 2, not 3: with EXTEND_INFILL_START_TO_WALL off the path no longer
        # opens with a lead-in segment, so everything after it shifts by one.
        self.assertTrue(np.allclose(legacy_path[2], np.asarray((1.0, 10.0), dtype=float)))
        self.assertTrue(np.allclose(swapped_path[2], np.asarray((9.0, 10.0), dtype=float)))

    def test_preview_uses_half_line_width_for_wall_gap_length(self):
        payload = {
            "contour_points": np.array([[0, 0], [6, 0], [6, 6], [0, 6]], dtype=float),
            "num_walls": 2,
            "line_width": 0.4,
            "infill_density": 30,
            "infill_pattern": "honeycomb",
        }

        with patch.object(main_module, "_build_full_toolpath", wraps=main_module._build_full_toolpath) as mocked:
            _build_pattern_preview(payload)

        self.assertTrue(mocked.called)
        self.assertAlmostEqual(
            float(mocked.call_args.kwargs["wall_gap_length"]),
            0.4 / 2.0 + WallSettings.CONNECTION_GAP_EXTRA,
            places=6,
        )

    def test_mesh_export_line_width_adds_configured_slicer_margin(self):
        preview_result = _PatternPreviewResult(
            contour_points=np.zeros((0, 2), dtype=float),
            path_points=np.array([[0.0, 0.0], [0.0, 5.0]], dtype=float),
            wall_centerlines=[],
            final_contour=None,
            line_width=0.4,
            fillet_radius=0.0,
            extrude_height=1.0,
            layer_height=0.2,
        )

        with patch.object(MeshSettings, "LINE_WIDTH_EXPORT_EXTRA", 0.05):
            self.assertAlmostEqual(_mesh_export_line_width(0.4), 0.45, places=6)
            vertices, faces = _build_preview_mesh(preview_result)

        self.assertGreater(len(vertices), 0)
        self.assertGreater(len(faces), 0)
        self.assertAlmostEqual(float(vertices[:, 0].max() - vertices[:, 0].min()), 0.45, places=3)
        self.assertAlmostEqual(preview_result.line_width, 0.4, places=6)

    def test_pattern_request_is_geometry_free(self):
        request = PatternRequest(
            num_walls=1,
            line_width=0.4,
            wall_spacing=0.0,
            fillet_radius=0.0,
            infill_density=50.0,
            infill_pattern="honeycomb",
            extrude_height=0.0,
        )

        self.assertFalse(hasattr(request, "contour_points"))
        self.assertFalse(hasattr(request, "stl_file"))
        self.assertFalse(hasattr(request, "title"))

    def test_global_pattern_plan_reuses_columns_for_layer_clipping(self):
        request = PatternRequest(
            num_walls=0,
            line_width=0.2,
            wall_spacing=0.0,
            fillet_radius=0.0,
            infill_density=30.0,
            infill_pattern="vertical",
            extrude_height=0.0,
        )
        slicer_output = self._slicer_output_with_layers([
            {
                "z": 0.0,
                "method": "exact",
                "contours": [
                    {
                        "id": 0,
                        "area": 16.0,
                        "is_largest": True,
                        "exterior": [[1.0, 1.0], [5.0, 1.0], [5.0, 5.0], [1.0, 5.0]],
                        "holes": [],
                    }
                ],
            },
            {
                "z": 0.2,
                "method": "exact",
                "contours": [
                    {
                        "id": 0,
                        "area": 16.0,
                        "is_largest": True,
                        "exterior": [[4.0, 1.0], [8.0, 1.0], [8.0, 5.0], [4.0, 5.0]],
                        "holes": [],
                    }
                ],
            },
        ])

        plan = build_global_pattern_plan(request, slicer_output)
        first = apply_global_pattern_to_layer(plan, slicer_output["layers"][0])
        second = apply_global_pattern_to_layer(plan, slicer_output["layers"][1])

        self.assertTrue(plan.raw_columns)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(plan.bounds_xy, (0.0, 0.0, 10.0, 8.0))
        self.assertTrue(np.any(first.path_points[:, 0] < 4.0))
        self.assertTrue(np.any(second.path_points[:, 0] > 5.0))

    def test_generate_pattern_builds_global_plan_once_for_multilayer_input(self):
        payload = {
            "request": {
                "num_walls": 0,
                "line_width": 0.2,
                "wall_spacing": 0.0,
                "fillet_radius": 0.0,
                "infill_density": 30.0,
                "infill_pattern": "vertical",
                "extrude_height": 0.4,
                "layer_height": 0.2,
            },
            "slicer_output": self._slicer_output_with_layers([
                self._layer(0.3, [[4.0, 1.0], [8.0, 1.0], [8.0, 5.0], [4.0, 5.0]], contour_id=1),
                self._layer(0.1, [[1.0, 1.0], [5.0, 1.0], [5.0, 5.0], [1.0, 5.0]], contour_id=0),
            ]),
        }

        with patch.object(main_module, "build_global_pattern_plan", wraps=main_module.build_global_pattern_plan) as mocked:
            result = generate_pattern(payload)

        self.assertEqual(mocked.call_count, 1)
        self.assertEqual([layer["z"] for layer in result["layers"]], [0.1, 0.3])
        self.assertNotAlmostEqual(
            float(np.mean(result["layers"][0]["path_points"][:, 0])),
            float(np.mean(result["layers"][1]["path_points"][:, 0])),
            places=3,
        )

    def test_multilayer_mesh_uses_layer_z_as_center(self):
        request = PatternRequest(
            num_walls=0,
            line_width=0.2,
            wall_spacing=0.0,
            fillet_radius=0.0,
            infill_density=30.0,
            infill_pattern="vertical",
            extrude_height=0.6,
            layer_height=0.2,
        )
        slicer_output = self._slicer_output_with_layers([
            self._layer(0.1, [[1.0, 1.0], [5.0, 1.0], [5.0, 5.0], [1.0, 5.0]], contour_id=0),
            self._layer(0.3, [[1.5, 1.0], [5.5, 1.0], [5.5, 5.0], [1.5, 5.0]], contour_id=1),
            self._layer(0.5, [[2.0, 1.0], [6.0, 1.0], [6.0, 5.0], [2.0, 5.0]], contour_id=2),
        ])
        plan = build_global_pattern_plan(request, slicer_output)
        previews = [
            apply_global_pattern_to_layer(plan, layer)
            for layer in slicer_output["layers"]
        ]
        previews = [preview for preview in previews if preview is not None]

        vertices, faces = _build_multilayer_mesh(previews)

        self.assertGreater(len(vertices), 0)
        self.assertGreater(len(faces), 0)
        self.assertAlmostEqual(float(vertices[:, 2].min()), 0.0, places=6)
        self.assertAlmostEqual(float(vertices[:, 2].max()), 0.6, places=6)

    def test_multilayer_top_and_bottom_layers_are_marked_solid(self):
        payload = {
            "request": {
                "num_walls": 1,
                "line_width": 0.25,
                "wall_spacing": 0.0,
                "fillet_radius": 0.0,
                "infill_density": 35.0,
                "infill_pattern": "vertical",
                "extrude_height": 0.8,
                "layer_height": 0.2,
                "bottom_layers": 1,
                "top_layers": 1,
            },
            "slicer_output": self._slicer_output_with_layers([
                self._layer(0.1, [[1.0, 1.0], [6.0, 1.0], [6.0, 6.0], [1.0, 6.0]], contour_id=0),
                self._layer(0.3, [[1.2, 1.1], [6.2, 1.1], [6.2, 6.1], [1.2, 6.1]], contour_id=1),
                self._layer(0.5, [[1.4, 1.2], [6.4, 1.2], [6.4, 6.2], [1.4, 6.2]], contour_id=2),
                self._layer(0.7, [[1.6, 1.3], [6.6, 1.3], [6.6, 6.3], [1.6, 6.3]], contour_id=3),
            ]),
        }

        result = generate_pattern(payload)

        self.assertEqual(
            [layer["mesh_kind"] for layer in result["layers"]],
            ["solid", "infill", "infill", "solid"],
        )

    @unittest.skipUnless(HAS_CAD_BACKEND, "needs the optional CAD backend: organic-infill-generator[cad]")
    def test_step_file_exports_single_preview_directly(self):
        payload = {
            "contour_points": [[0.0, 0.0], [8.0, 0.0], [8.0, 6.0], [0.0, 6.0]],
            "num_walls": 0,
            "line_width": 0.3,
            "wall_spacing": 0.0,
            "fillet_radius": 0.0,
            "infill_density": 25.0,
            "infill_pattern": "vertical",
            "extrude_height": 0.4,
            "layer_height": 0.2,
            "step_file": True,
            "title": "direct_step_single",
        }

        previous_results_dir = PathSettings.RESULTS_DIR
        with tempfile.TemporaryDirectory() as tmp:
            PathSettings.RESULTS_DIR = tmp
            try:
                result = generate_pattern(payload)
            finally:
                PathSettings.RESULTS_DIR = previous_results_dir

            self.assertTrue(os.path.exists(result["step_path"]))
            self.assertGreater(os.path.getsize(result["step_path"]), 0)
            self.assertTrue(result["step_path"].endswith(".step"))

    @unittest.skipUnless(HAS_CAD_BACKEND, "needs the optional CAD backend: organic-infill-generator[cad]")
    def test_step_file_exports_multilayer_stack_directly(self):
        payload = {
            "request": {
                "num_walls": 0,
                "line_width": 0.25,
                "wall_spacing": 0.0,
                "fillet_radius": 0.0,
                "infill_density": 30.0,
                "infill_pattern": "vertical",
                "extrude_height": 0.4,
                "layer_height": 0.2,
            },
            "slicer_output": self._slicer_output_with_layers([
                self._layer(0.1, [[1.0, 1.0], [5.0, 1.0], [5.0, 5.0], [1.0, 5.0]], contour_id=0),
                self._layer(0.3, [[1.5, 1.0], [5.5, 1.0], [5.5, 5.0], [1.5, 5.0]], contour_id=1),
            ]),
            "options": {
                "step_file": True,
                "title": "direct_step_stack",
            },
        }

        previous_results_dir = PathSettings.RESULTS_DIR
        with tempfile.TemporaryDirectory() as tmp:
            PathSettings.RESULTS_DIR = tmp
            try:
                result = generate_pattern(payload)
            finally:
                PathSettings.RESULTS_DIR = previous_results_dir

            self.assertEqual(len(result["layers"]), 2)
            self.assertTrue(os.path.exists(result["step_path"]))
            self.assertGreater(os.path.getsize(result["step_path"]), 0)

    def test_trapezoid_multilayer_part_generates_stl(self):
        layers = [
            self._layer(0.1 + 0.2 * index, [[0.0, 0.0], [10.0 - 0.1 * index, 0.0], [7.0, 8.0], [0.0, 8.0]])
            for index in range(5)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            result = generate_pattern(
                {
                    "num_walls": 1,
                    "line_width": 0.4,
                    "wall_spacing": 0.0,
                    "fillet_radius": 0.0,
                    "infill_density": 40.0,
                    "infill_pattern": "honeycomb",
                    "extrude_height": 1.0,
                    "layer_height": 0.2,
                    "bottom_layers": 1,
                    "top_layers": 0,
                },
                self._slicer_output_with_layers(layers),
                {"stl_file": True, "output_dir": temp_dir},
            )

            self.assertEqual(len(result["layers"]), 5)
            self.assertTrue(os.path.exists(result["stl_path"]))
            self.assertEqual(result["layers"][0]["mesh_kind"], "solid")
            self.assertEqual(result["layers"][-1]["mesh_kind"], "infill")

    def test_slicer_generation_uses_largest_contour_and_ignores_holes(self):
        payload = {
            "request": {
                "num_walls": 0,
                "line_width": 0.2,
                "wall_spacing": 0.0,
                "fillet_radius": 0.0,
                "infill_density": 30.0,
                "infill_pattern": "vertical",
                "extrude_height": 0.0,
            },
            "slicer_output": self._slicer_output_with_layers([
                {
                    "z": 0.0,
                    "method": "exact",
                    "contours": [
                        {
                            "id": 0,
                            "area": 1.0,
                            "is_largest": False,
                            "exterior": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                            "holes": [],
                        },
                        {
                            "id": 1,
                            "area": 25.0,
                            "is_largest": True,
                            "exterior": [[2.0, 2.0], [7.0, 2.0], [7.0, 7.0], [2.0, 7.0]],
                            "holes": [[[3.0, 3.0], [4.0, 3.0], [4.0, 4.0], [3.0, 4.0]]],
                        },
                    ],
                }
            ]),
        }

        result = generate_pattern(payload)

        self.assertEqual(len(result["layers"]), 1)
        self.assertGreater(len(result["path_points"]), 0)
        self.assertGreaterEqual(float(np.min(result["path_points"][:, 0])), 2.0)

    def test_density_derivation_defaults(self):
        request = resolve_pattern_request({
            "contour_points": np.array([[0, 0], [5, 0], [5, 5], [0, 5]]),
            "line_width": 0.4,
            "infill_density": 50,
            "infill_pattern": "honeycomb",
        })
        self.assertAlmostEqual(request.base_line_distance, 0.8, places=3)
        expected_line_distance = 0.8 * PatternCatalog.DISTANCE_FACTORS["honeycomb"]
        self.assertAlmostEqual(request.line_distance, expected_line_distance, places=3)
        self.assertAlmostEqual(
            request.infill_spacing,
            expected_line_distance + 0.4 + 2.0 * request.wall_spacing,
            places=3,
        )
        self.assertAlmostEqual(request.pattern_distance_factor, PatternCatalog.DISTANCE_FACTORS["honeycomb"], places=3)
        self.assertAlmostEqual(request.pattern_amplitude, expected_line_distance, places=3)
        self.assertAlmostEqual(request.pattern_step, expected_line_distance * PatternCatalog.HEXAGON_STEP_RATIO, places=3)
        self.assertAlmostEqual(request.estimated_density, 50.0, places=3)

    def test_infill_distance_rule_for_vertical_and_sparse_patterns(self):
        vertical = resolve_pattern_request({
            "line_width": 0.4,
            "infill_density": 100,
            "infill_pattern": "vertical",
        })

        self.assertAlmostEqual(vertical.line_distance, 0.4, places=6)
        self.assertAlmostEqual(vertical.pattern_distance_factor, 1.0, places=6)

        expected_factors = {
            "honeycomb": PatternCatalog.DISTANCE_FACTORS["honeycomb"],
            "triangular": PatternCatalog.DISTANCE_FACTORS["triangular"],
            "sinusoidal": PatternCatalog.DISTANCE_FACTORS["sinusoidal"],
        }
        for pattern in ("honeycomb", "triangular", "sinusoidal"):
            with self.subTest(pattern=pattern):
                resolved = resolve_pattern_request({
                    "line_width": 0.4,
                    "infill_density": 100,
                    "infill_pattern": pattern,
                })
                self.assertAlmostEqual(resolved.line_distance, 0.4 * expected_factors[pattern], places=6)
                self.assertAlmostEqual(resolved.pattern_distance_factor, expected_factors[pattern], places=6)

    def test_honeycomb_cell_proportion_is_not_overscaled(self):
        resolved = resolve_pattern_request({
            "line_width": 0.4,
            "infill_density": 100,
            "infill_pattern": "honeycomb",
        })

        self.assertAlmostEqual(resolved.pattern_amplitude, resolved.line_distance, places=6)
        self.assertAlmostEqual(resolved.pattern_step, resolved.line_distance * PatternCatalog.HEXAGON_STEP_RATIO, places=6)

    def test_frankenstein_uses_mixed_sequence_defaults_without_special_override(self):
        resolved = resolve_pattern_request({
            "line_width": 0.4,
            "infill_density": 100,
            "infill_pattern": "frankenstein",
        })

        self.assertIn("HH", resolved.pattern_sequence)
        self.assertIn("HHI", resolved.pattern_sequence)
        self.assertAlmostEqual(resolved.pattern_distance_factor, PatternCatalog.DISTANCE_FACTORS["triangular"], places=6)
        self.assertAlmostEqual(resolved.pattern_step, resolved.pattern_amplitude, places=6)

    def test_wall_spacing_is_public_and_affects_spacing_resolution(self):
        """
        What is being pinned is the contract, not a calibration. This used to
        assert a literal 1.6 mm amplitude, which was the honeycomb distance
        factor of the day; that factor is deliberately tunable (it moved from
        2.0 to 1.5, see ``PatternCatalog.DISTANCE_FACTORS``) and freezing it
        here meant a recalibration looked like a regression.
        """
        def resolve(wall_spacing):
            return resolve_pattern_request({
                "contour_points": np.array([[0, 0], [5, 0], [5, 5], [0, 5]]),
                "line_width": 0.4,
                "wall_spacing": wall_spacing,
                "infill_density": 50,
                "infill_pattern": "honeycomb",
            })

        request = resolve(0.15)
        self.assertAlmostEqual(request.wall_spacing, 0.15, places=3)
        self.assertEqual(request.pattern_sequence, ["HH", "HHI"])
        # The gap is counted once on each side of a column.
        self.assertAlmostEqual(
            request.infill_spacing,
            request.pattern_amplitude + 2 * 0.15 + 0.4,
            places=6,
        )
        # And it is public: moving it moves the spacing by twice the delta,
        # leaving the pattern's own amplitude alone.
        wider = resolve(0.25)
        self.assertAlmostEqual(
            wider.infill_spacing - request.infill_spacing, 2 * 0.10, places=6,
        )
        self.assertAlmostEqual(
            wider.pattern_amplitude, request.pattern_amplitude, places=9,
        )

    def test_negative_wall_spacing_is_allowed_for_wall_overlap(self):
        request = resolve_pattern_request({
            "contour_points": np.array([[0, 0], [5, 0], [5, 5], [0, 5]]),
            "wall_spacing": -0.05,
            "line_width": 0.4,
            "infill_pattern": "sinusoidal",
        })
        self.assertAlmostEqual(request.wall_spacing, -0.05, places=3)
        self.assertGreater(request.infill_spacing, 0.0)
        self.assertAlmostEqual(request.infill_spacing, request.line_distance + request.line_width - 0.1, places=3)

    def test_layered_extrusion_parameters_are_resolved(self):
        request = resolve_pattern_request({
            "contour_points": np.array([[0, 0], [5, 0], [5, 5], [0, 5]]),
            "line_width": 0.4,
            "infill_pattern": "honeycomb",
            "extrude_height": 2.0,
            "layer_height": 0.2,
            "bottom_layers": 2,
            "top_layers": 3,
        })

        self.assertAlmostEqual(request.extrude_height, 2.0, places=6)
        self.assertAlmostEqual(request.layer_height, 0.2, places=6)
        self.assertEqual(request.bottom_layers, 2)
        self.assertEqual(request.top_layers, 3)

    def test_layered_extrusion_rejects_solid_height_above_total(self):
        with self.assertRaises(ValueError):
            resolve_pattern_request({
                "contour_points": np.array([[0, 0], [5, 0], [5, 5], [0, 5]]),
                "line_width": 0.4,
                "infill_pattern": "honeycomb",
                "extrude_height": 0.5,
                "layer_height": 0.2,
                "bottom_layers": 2,
                "top_layers": 2,
            })

    def test_wall_spacing_cannot_collapse_centerline_step(self):
        with self.assertRaises(ValueError):
            resolve_pattern_request({
            "contour_points": np.array([[0, 0], [5, 0], [5, 5], [0, 5]]),
                "line_width": 0.4,
                "wall_spacing": -0.4,
                "infill_pattern": "honeycomb",
            })

    def test_mixed_vertical_columns_use_narrow_line_width_based_spacing(self):
        layout = iter_column_layout(
            minx=0.0,
            maxx=6.0,
            pattern_sequence=["S", "V", "I", "V"],
            infill_spacing=1.0,
            line_width=0.2,
            column_clearance=0.05,
        )

        spans = [span for _, _, span in layout[:4]]
        centers = [x for x, _, _ in layout[:4]]

        self.assertAlmostEqual(spans[0], 1.0, places=3)
        self.assertAlmostEqual(spans[1], 0.3, places=3)
        self.assertAlmostEqual(spans[2], 1.0, places=3)
        self.assertAlmostEqual(spans[3], 0.3, places=3)
        self.assertAlmostEqual(centers[1] - centers[0], 0.65, places=3)
        self.assertAlmostEqual(centers[2] - centers[1], 0.65, places=3)
        self.assertAlmostEqual(centers[3] - centers[2], 0.65, places=3)

    def test_standalone_vertical_keeps_global_spacing(self):
        layout = iter_column_layout(
            minx=0.0,
            maxx=6.0,
            pattern_sequence=["V"],
            infill_spacing=1.0,
            line_width=0.2,
            column_clearance=0.05,
        )

        spans = [span for _, _, span in layout[:2]]
        centers = [x for x, _, _ in layout[:2]]

        self.assertAlmostEqual(spans[0], 1.0, places=3)
        self.assertAlmostEqual(spans[1], 1.0, places=3)
        self.assertAlmostEqual(centers[1] - centers[0], 1.0, places=3)

    def test_sinusoidal_starts_from_lateral_side(self):
        columns = build_infill_columns(
            infill_boundary=box(0.0, 0.0, 2.0, 4.0),
            pattern_sequence=["SIN"],
            infill_spacing=0.8,
            pattern_amplitude=0.4,
            pattern_step=0.4,
            density_profile=None,
            line_width=0.1,
            column_clearance=WallSettings.SPACING,
        )

        self.assertTrue(columns)
        coords = np.asarray(columns[0].coords, dtype=float)
        x_center = float(np.mean(coords[:, 0]))
        self.assertAlmostEqual(abs(float(coords[0, 0]) - x_center), 0.2, places=2)

    def test_sinusoidal_last_unit_is_not_squished_to_y_end(self):
        from organic_infill_generator.infill_generator.patterns.pattern_sinusoidal import SinusoidalPattern

        pattern = SinusoidalPattern()
        points = pattern.generate(
            x_center=0.0,
            y_start=0.0,
            y_end=1.0,
            pattern_amplitude=0.6,
            pattern_step=0.6,
            density_profile=None,
            line_width=0.1,
        )
        ys = np.asarray(points, dtype=float)[:, 1]
        self.assertGreater(float(ys.max()), 1.0)
        last_increment = float(ys[-1] - ys[-2])
        expected_increment = max(
            0.6 / SinusoidalSettings.STEPS_PER_PATTERN_STEP,
            SinusoidalSettings.MIN_INCREMENT,
        )
        self.assertAlmostEqual(last_increment, expected_increment, places=6)

    def test_snake_assembler_uses_provided_columns_without_reclipping(self):
        infill_boundary = box(0.0, 0.0, 5.0, 5.0)
        raw_columns = [
            LineString([(1.0, -2.0), (1.0, 7.0)]),
            LineString([(3.0, -2.0), (3.0, 7.0)]),
        ]
        columns_union, _, _, _, _ = assemble_eulerian_path(
            infill_boundary=infill_boundary,
            columns=raw_columns,
            line_width=0.0,
        )

        self.assertAlmostEqual(
            columns_union.length,
            sum(column.length for column in raw_columns),
            places=3,
        )

    def test_forbidden_area_split_keeps_column_fragments(self):
        infill_boundary = box(0.0, 0.0, 10.0, 10.0)
        columns = [
            LineString([(2.0, 0.5), (2.0, 9.5)]),
            LineString([(3.0, 0.5), (2.0, 5.0), (3.0, 9.5)]),
        ]

        _, _, _, _, ordered = assemble_eulerian_path(
            infill_boundary=infill_boundary,
            columns=columns,
            line_width=1.0,
        )

        right_side_points = [point for point in ordered if point[0] > 2.5]
        self.assertTrue(right_side_points)
        self.assertLess(min(point[1] for point in right_side_points), 1.5)
        self.assertGreater(max(point[1] for point in right_side_points), 8.5)

    def test_clipped_columns_keep_half_line_width_clearance_from_wall(self):
        infill_boundary = box(0.0, 0.0, 10.0, 10.0)
        raw_columns = [
            LineString([(0.1, -1.0), (0.1, 11.0)]),
            LineString([(5.0, -1.0), (5.0, 11.0)]),
        ]

        clipped = clip_and_reconnect_columns(
            raw_columns,
            infill_boundary,
            line_width=1.0,
        )

        self.assertEqual(len(clipped), 1)
        self.assertGreaterEqual(clipped[0].distance(infill_boundary.boundary), 0.5 - 1e-6)

    def test_clipped_concave_fragments_choose_one_connector_route(self):
        infill_boundary = Polygon([
            (0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (5.0, 8.0),
            (5.0, 3.0), (3.0, 3.0), (3.0, 8.0), (0.0, 8.0),
        ])
        raw_column = LineString([(4.0, -1.0), (4.0, 9.0)])

        clipped = clip_and_reconnect_columns(
            [raw_column],
            infill_boundary,
            line_width=1.0,
        )

        self.assertEqual(len(clipped), 1)
        line = clipped[0]
        coords = np.asarray(line.coords, dtype=float)
        self.assertLess(line.length, 8.5)
        self.assertFalse(
            np.any(
                (np.isclose(coords[:, 0], 3.5) | np.isclose(coords[:, 0], 4.5))
                & (coords[:, 1] > 3.5)
            )
        )
        self.assertGreaterEqual(clipped[0].distance(infill_boundary.boundary), 0.5 - 1e-6)

    def test_line_width_clipping_reconnects_fragments_without_extra_direct_bridge(self):
        infill_boundary = Point(0.0, 0.0).buffer(5.0, quad_segs=64)
        raw_column = LineString([(0.0, -6.0), (4.7, 0.0), (0.0, 6.0)])

        clipped = clip_and_reconnect_columns(
            [raw_column],
            infill_boundary,
            line_width=1.0,
        )

        self.assertEqual(len(clipped), 1)
        self.assertGreater(clipped[0].length, 10.0)
        self.assertLess(clipped[0].length, raw_column.length)
        self.assertGreaterEqual(clipped[0].distance(infill_boundary.boundary), 0.5 - 1e-6)
        coords = np.asarray(clipped[0].coords, dtype=float)
        self.assertGreater(len(coords), 3)
        self.assertLess(clipped[0].length, raw_column.length * 0.75)

    def test_plot_legend_is_compact_and_title_includes_distance(self):
        preview_result = _build_pattern_preview({
            "contour_points": np.array([[0, 0], [5, 0], [5, 5], [0, 5]]),
            "num_walls": 3,
            "line_width": 0.4,
            "infill_density": 50,
            "infill_pattern": "honeycomb",
        })
        fig, ax = plt.subplots(figsize=(6, 5))
        plot_pattern_result(preview_result, ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        plt.close(fig)

        self.assertEqual(
            labels,
            ["Centerline", "Line width", "3D Fillet Preview", "Start", "Infill Start", "End"],
        )
        self.assertIn("Distance=", ax.get_title())
        self.assertIn("Fillet>=", ax.get_title())

if __name__ == "__main__":
    unittest.main()

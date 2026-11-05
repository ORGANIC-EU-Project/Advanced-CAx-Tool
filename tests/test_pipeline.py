from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh
from _fixtures import write_box_stl, write_drafted_stl

from organic_infill_generator import PipelineConfig, run_pipeline
from organic_infill_generator.infill_generator.generator import TOOLPATH_CSV_COLUMNS, _build_layer_mesh
from organic_infill_generator.slicer import ClosureResult

HAS_CAD_BACKEND = importlib.util.find_spec("cadquery") is not None


class TestRunPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.dir = Path(self._temp.name)
        self.stl = write_drafted_stl(self.dir / "drafted.stl")
        self.config = PipelineConfig(
            layer_height=6.0,
            line_width=6.0,
            infill_density=40.0,
            infill_pattern="honeycomb",
            num_walls=1,
            wall_spacing=0.2,
            fillet_radius=6.0,
            title="drafted",
            output_dir=str(self.dir / "out"),
        )

    def test_a_stl_part_goes_all_the_way_to_the_outputs(self):
        result = run_pipeline(self.stl, self.config)

        self.assertEqual(result.slicer_result.output.closed_layers, 6)
        self.assertEqual(len(result.previews), 6)
        for path in (result.contours_path, result.toolpath_csv_path, result.output_stl_path, result.summary_path):
            self.assertTrue(Path(path).exists(), path)
            self.assertGreater(Path(path).stat().st_size, 0, path)
        self.assertIsNone(result.output_step_path)

    def test_toolpath_csv_holds_every_layer_in_order(self):
        result = run_pipeline(self.stl, self.config)

        with open(result.toolpath_csv_path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(tuple(rows[0].keys()), TOOLPATH_CSV_COLUMNS)
        layers = [int(row["layer"]) for row in rows]
        self.assertEqual(sorted(set(layers)), [1, 2, 3, 4, 5, 6])
        self.assertEqual(layers, sorted(layers))
        self.assertEqual({row["path_type"] for row in rows}, {"infill", "wall"})
        heights = sorted({float(row["z"]) for row in rows})
        self.assertEqual(heights, [3.0, 9.0, 15.0, 21.0, 27.0, 33.0])

    def test_output_stl_is_one_closed_body_per_layer(self):
        result = run_pipeline(self.stl, self.config)

        # process=False: merging the whole stack would weld layers that touch.
        mesh = trimesh.load(result.output_stl_path, process=False)

        self.assertAlmostEqual(float(mesh.bounds[0, 2]), 0.0, places=3)
        self.assertAlmostEqual(float(mesh.bounds[1, 2]), 36.0, places=3)
        layer_faces = 0
        for preview in result.previews:
            vertices, faces = _build_layer_mesh(preview)
            layer_faces += len(faces)
            body = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
            self.assertTrue(body.is_watertight, preview.z)
            self.assertAlmostEqual(float(body.extents[2]), 6.0, places=3)
        self.assertEqual(len(mesh.faces), layer_faces)

    def test_bead_volume_matches_the_path(self):
        # Lines far apart: a cell pattern shares walls, so its beads overlap.
        self.config.infill_pattern = "vertical"
        self.config.infill_density = 25.0
        self.config.num_walls = 0
        self.config.fillet_radius = 0.0

        result = run_pipeline(self.stl, self.config)

        volume = float(trimesh.load(result.output_stl_path, process=False).volume)
        length = result.summary["validation"]["total_path_length_mm"]
        self.assertAlmostEqual(abs(volume), length * 6.0 * 6.0, delta=length * 36.0 * 0.10)

    def test_solid_layers_cover_the_part(self):
        # The raster of a solid layer must fit a whole number of beads: see
        # test_solid_layer_that_no_raster_fits_is_refused for when it does not.
        panel = write_box_stl(self.dir / "panel.stl", (126.0, 80.0, 36.0))
        self.config.bottom_layers = 1
        self.config.top_layers = 1
        result = run_pipeline(panel, self.config)

        kinds = [preview.mesh_kind for preview in result.previews]
        self.assertEqual(kinds, ["solid", "infill", "infill", "infill", "infill", "solid"])
        with open(result.toolpath_csv_path, newline="", encoding="utf-8") as handle:
            types = {(row["layer"], row["path_type"]) for row in csv.DictReader(handle)}
        self.assertIn(("1", "solid"), types)
        self.assertIn(("6", "solid"), types)

    def test_solid_layer_that_no_raster_fits_is_refused(self):
        """One bead width cannot fill every width: refused, not over-extruded."""
        panel = write_box_stl(self.dir / "panel.stl", (126.2, 80.0, 36.0))
        self.config.bottom_layers = 1

        with self.assertRaisesRegex(ValueError, "Solid raster cannot touch"):
            run_pipeline(panel, self.config)

    def test_summary_reports_a_valid_toolpath(self):
        result = run_pipeline(self.stl, self.config)

        summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["slicer"]["failed_layers"], 0)
        self.assertEqual(summary["validation"]["total_jumps"], 0)
        self.assertEqual(summary["validation"]["layers_with_toolpath"], 6)
        self.assertGreater(summary["validation"]["total_path_length_mm"], 0.0)
        self.assertTrue(summary["process"]["fillet_applied"])
        self.assertEqual(summary["input_file"], "drafted.stl")
        self.assertIn("slice_and_close_contours", summary["timings_s"])

    def test_identical_runs_give_identical_paths(self):
        first = run_pipeline(self.stl, self.config)
        second = run_pipeline(self.stl, self.config)

        for one, other in zip(first.previews, second.previews):
            np.testing.assert_array_equal(one.path_points, other.path_points)

    def test_exports_can_be_switched_off(self):
        self.config.export_csv = False
        self.config.export_stl = False

        result = run_pipeline(self.stl, self.config)

        self.assertIsNone(result.toolpath_csv_path)
        self.assertIsNone(result.output_stl_path)

    def test_every_pattern_runs(self):
        for pattern in ("vertical", "serrated", "serrated_inverted", "sinusoidal", "inclined",
                        "honeycomb", "frankenstein", "hilbert", "diamond", "triangular"):
            with self.subTest(pattern=pattern):
                self.config.infill_pattern = pattern
                self.config.export_stl = False

                result = run_pipeline(self.stl, self.config)

                self.assertEqual(result.summary["validation"]["total_jumps"], 0)

    def test_unknown_pattern_is_refused(self):
        self.config.infill_pattern = "nonsense"
        with self.assertRaises(ValueError):
            run_pipeline(self.stl, self.config)

    def test_bead_wider_than_the_part_is_refused_before_any_geometry(self):
        small = write_box_stl(self.dir / "small.stl", (10.0, 10.0, 6.0))
        self.config.line_width = 40.0
        with self.assertRaises(ValueError):
            run_pipeline(small, self.config)

    def test_unsupported_input_is_refused(self):
        other = self.dir / "part.obj"
        other.write_text("", encoding="utf-8")
        with self.assertRaises(ValueError):
            run_pipeline(other, self.config)

    def test_missing_input_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            run_pipeline(self.dir / "missing.stl", self.config)

    def test_part_with_no_closed_contour_stops_with_a_clear_error(self):
        class NothingCloses:
            def close(self, segments: np.ndarray, model_diag: float) -> ClosureResult:
                return ClosureResult(open_chains=[np.asarray(segments)[0]])

        self.config.contour_closer = NothingCloses()
        with self.assertRaisesRegex(RuntimeError, "no closed contours"):
            run_pipeline(self.stl, self.config)


@unittest.skipUnless(HAS_CAD_BACKEND, "needs the optional CAD backend: organic-infill-generator[cad]")
class TestStepRoundTrip(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.dir = Path(self._temp.name)

    def _write_step(self) -> Path:
        import cadquery as cq

        path = self.dir / "plate.step"
        cq.exporters.export(cq.Workplane("XY").box(120.0, 80.0, 12.0), str(path))
        return path

    def test_step_to_stl_keeps_the_part(self):
        from organic_infill_generator.cadquery_service import step_to_stl

        target = self.dir / "plate.stl"
        report = step_to_stl(self._write_step(), target)

        self.assertEqual(report["operation"], "STEP -> STL")
        mesh = trimesh.load(target)
        np.testing.assert_allclose(mesh.extents, [120.0, 80.0, 12.0], atol=1e-3)
        self.assertAlmostEqual(float(mesh.volume), 120.0 * 80.0 * 12.0, places=0)

    def test_a_step_part_goes_through_the_whole_pipeline_and_back_to_step(self):
        result = run_pipeline(self._write_step(), PipelineConfig(
            layer_height=6.0,
            infill_density=40.0,
            output_dir=str(self.dir / "out"),
            title="plate",
            export_step=True,
        ))

        self.assertEqual(result.slicer_result.output.closed_layers, 2)
        self.assertTrue(result.stl_path.exists())
        self.assertGreater(Path(result.output_step_path).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

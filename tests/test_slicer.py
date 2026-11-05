from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh
from _fixtures import write_box_stl, write_drafted_stl, write_tube_stl

from organic_infill_generator.slicer import ClosureResult, save_contours, slice_contours


class TestSliceContours(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.dir = Path(self._temp.name)

    def test_box_gives_one_exact_contour_per_layer(self):
        stl = write_box_stl(self.dir / "box.stl", (120.0, 80.0, 12.0))

        output = slice_contours(str(stl), layer_height=3.0, parallel=False).output

        self.assertEqual(output.total_layers, 4)
        self.assertEqual(output.closed_layers, 4)
        self.assertEqual(output.failed_layers, 0)
        self.assertEqual([layer.z for layer in output.layers], [1.5, 4.5, 7.5, 10.5])
        for layer in output.layers:
            self.assertEqual(layer.method, "exact")
            self.assertEqual(len(layer.contours), 1)
            self.assertTrue(layer.contours[0].is_largest)
            self.assertAlmostEqual(layer.contours[0].area, 9600.0, places=3)

    def test_a_plane_above_the_part_is_not_a_layer(self):
        stl = write_box_stl(self.dir / "box.stl", (30.0, 20.0, 10.5))

        output = slice_contours(str(stl), layer_height=3.0, parallel=False).output

        self.assertEqual([layer.z for layer in output.layers], [1.5, 4.5, 7.5])
        self.assertEqual(output.closed_layers, output.total_layers)

    def test_part_is_moved_to_the_origin(self):
        mesh = trimesh.creation.box(extents=(20.0, 10.0, 6.0))
        mesh.apply_translation((500.0, -300.0, 77.0))
        stl = self.dir / "offset.stl"
        mesh.export(stl)

        output = slice_contours(str(stl), layer_height=2.0, parallel=False).output

        np.testing.assert_allclose(output.model_bounds.min, [0.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(output.model_bounds.size, [20.0, 10.0, 6.0], atol=1e-5)

    def test_drafted_part_area_shrinks_as_the_exact_section(self):
        stl = write_drafted_stl(self.dir / "drafted.stl")

        output = slice_contours(str(stl), layer_height=6.0, parallel=False).output

        self.assertEqual(output.total_layers, 6)
        for layer in output.layers:
            t = layer.z / 36.0
            width = 150.0 - (40.0 + 18.0) * t
            depth = 100.0 - 20.0 * t
            self.assertAlmostEqual(layer.contours[0].area, width * depth, places=3)

    def test_a_closed_stl_is_not_reported_as_open(self):
        stl = write_drafted_stl(self.dir / "drafted.stl")

        with self.assertLogs("organic_infill_generator.slicer.config", level="INFO") as captured:
            slice_contours(str(stl), layer_height=6.0, parallel=False)

        self.assertIn("Mesh is watertight", [record.getMessage() for record in captured.records])

    def test_hole_is_recovered_as_a_hole_of_its_shell(self):
        stl = write_tube_stl(self.dir / "tube.stl", outer=50.0, inner=25.0, height=10.0)

        output = slice_contours(str(stl), layer_height=5.0, parallel=False).output

        for layer in output.layers:
            self.assertEqual(len(layer.contours), 1)
            self.assertEqual(len(layer.contours[0].holes), 1)
            expected = np.pi * (50.0 ** 2 - 25.0 ** 2)
            self.assertAlmostEqual(layer.contours[0].area, expected, delta=expected * 0.01)

    def test_missing_faces_are_reported_and_never_invented(self):
        mesh = trimesh.creation.box(extents=(40.0, 40.0, 10.0))
        keep = np.abs(mesh.face_normals[:, 0]) < 0.5  # drops the two X side walls
        mesh.update_faces(keep)
        stl = self.dir / "open.stl"
        mesh.export(stl)

        output = slice_contours(str(stl), layer_height=5.0, parallel=False).output

        self.assertEqual(output.failed_layers, output.total_layers)
        self.assertEqual(output.closed_layers, 0)
        for layer in output.layers:
            self.assertEqual(layer.method, "failed")
            self.assertEqual(layer.contours, [])
            self.assertEqual(len(layer.open_lines), 2)

    def test_parallel_and_sequential_results_match(self):
        stl = write_drafted_stl(self.dir / "drafted.stl")

        sequential = slice_contours(str(stl), layer_height=6.0, parallel=False).output
        parallel = slice_contours(str(stl), layer_height=6.0, parallel=True, num_workers=2).output

        self.assertEqual(sequential, parallel)

    def test_contour_closer_is_replaceable(self):
        class NothingCloses:
            calls = 0

            def close(self, segments: np.ndarray, model_diag: float) -> ClosureResult:
                type(self).calls += 1
                self.model_diag = model_diag
                return ClosureResult(open_chains=[np.asarray(segments)[0]])

        closer = NothingCloses()
        stl = write_box_stl(self.dir / "box.stl", (30.0, 40.0, 4.0))

        output = slice_contours(str(stl), layer_height=2.0, contour_closer=closer, parallel=False).output

        self.assertEqual(NothingCloses.calls, 2)
        self.assertEqual(output.failed_layers, 2)
        self.assertAlmostEqual(closer.model_diag, np.linalg.norm([30.0, 40.0, 4.0]), places=4)

    def test_contours_are_saved_as_json(self):
        stl = write_box_stl(self.dir / "box.stl", (30.0, 20.0, 4.0))
        result = slice_contours(str(stl), layer_height=2.0, parallel=False)
        target = self.dir / "contours.json"

        save_contours(result, str(target))

        data = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(data["total_layers"], 2)
        self.assertGreaterEqual(len(data["layers"][0]["contours"][0]["exterior"]), 4)

    def test_missing_file_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            slice_contours(str(self.dir / "nothing.stl"))


if __name__ == "__main__":
    unittest.main()

"""Layers that offset to the same walls reuse one toolpath computation: the
result must be exactly what computing every layer gives, at each layer's own
z and with its own contour, with no arrays shared between layers."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

import organic_infill_generator.infill_generator.generator as generator
from organic_infill_generator.infill_generator.generator import (
    _build_layer_outputs,
    apply_global_pattern_to_layer,
    build_global_pattern_plan,
)

REQUEST = {
    "num_walls": 1, "line_width": 2.0, "wall_spacing": 0.0, "fillet_radius": 2.0,
    "infill_density": 40.0, "infill_pattern": "honeycomb",
    "extrude_height": 5.0, "layer_height": 1.0, "bottom_layers": 1, "top_layers": 1,
}


def _slicer_output(exteriors):
    return {
        "layer_height": 1.0,
        "model_bounds": {"min": [0.0, 0.0, 0.0], "max": [60.0, 60.0, 5.0], "size": [60.0, 60.0, 5.0]},
        "layers": [
            {
                "z": index + 0.5, "method": "test", "open_lines": [],
                "contours": [{"id": 0, "area": 0.0, "is_largest": True, "exterior": exterior, "holes": []}],
            }
            for index, exterior in enumerate(exteriors)
        ],
        "total_layers": len(exteriors), "closed_layers": len(exteriors),
        "failed_layers": 0, "noise_layers": 0,
    }


def _square(inset):
    return [[inset, inset], [60.0 - inset, inset], [60.0 - inset, 60.0 - inset], [inset, 60.0 - inset]]


class TestLayerCache(unittest.TestCase):
    def _check(self, exteriors, expected_calls):
        slicer_output = _slicer_output(exteriors)
        plan = build_global_pattern_plan(REQUEST, slicer_output)
        reference = [
            apply_global_pattern_to_layer(plan, layer, title="t")
            for layer in slicer_output["layers"]
        ]
        # The toolpath assembly is the work the cache saves; count it.
        calls = []
        original = generator.assemble_eulerian_path

        def counted(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        with mock.patch.object(generator, "assemble_eulerian_path", counted):
            cached = _build_layer_outputs(plan, slicer_output, title="t")

        self.assertEqual(len(calls), expected_calls)
        self.assertEqual(len(cached), len(reference))
        for got, want in zip(cached, reference):
            self.assertEqual(got.z, want.z)
            np.testing.assert_array_equal(got.contour_points, want.contour_points)
            self.assertTrue(got.final_contour.equals_exact(want.final_contour, 0.0))
            np.testing.assert_array_equal(np.asarray(got.path_points), np.asarray(want.path_points))
            for got_wall, want_wall in zip(got.wall_centerlines, want.wall_centerlines):
                np.testing.assert_array_equal(got_wall, want_wall)
            self.assertEqual(got.fillet_protected_indices, want.fillet_protected_indices)
            self.assertTrue(got.mesh_preview_geometry.equals_exact(want.mesh_preview_geometry, 0.0))
        kinds = [preview.mesh_kind for preview in cached]
        self.assertEqual(kinds, ["solid"] + ["infill"] * (len(cached) - 2) + ["solid"])
        return cached

    def test_identical_contours_are_computed_once(self):
        cached = self._check([_square(0.0)] * 5, expected_calls=1)
        self.assertIsNot(cached[0].path_points, cached[1].path_points)
        self.assertIsNot(cached[0].wall_centerlines, cached[1].wall_centerlines)

    def test_collinear_vertices_do_not_defeat_the_cache(self):
        # What the slicer really returns for a prism: one outline, with the
        # points where each plane cut the STL's triangles left on its edges.
        def with_collinear(y):
            return [[60.0, 60.0], [60.0, y], [60.0, 0.0], [0.0, 0.0], [0.0, 60.0]]

        cached = self._check([with_collinear(y) for y in (55.0, 40.0, 25.0, 10.0, 5.0)], expected_calls=1)
        self.assertEqual(cached[1].contour_points[1].tolist(), [60.0, 40.0])

    def test_tapered_contours_are_each_computed(self):
        self._check([_square(float(i)) for i in range(5)], expected_calls=5)


if __name__ == "__main__":
    unittest.main()

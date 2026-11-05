import unittest
from types import SimpleNamespace

from organic_infill_generator.infill_generator.preflight import (
    InfeasibleRequestError,
    preflight_pattern_request,
)


def _request(**overrides):
    values = {
        "extrude_height": 10.0,
        "layer_height": 1.0,
        "line_width": 2.0,
        "wall_spacing": 0.0,
        "num_walls": 1,
        "infill_density": 20.0,
        "infill_spacing": 5.0,
        "infill_pattern": "honeycomb",
        "estimated_density": 20.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestPreflight(unittest.TestCase):
    def test_valid_request_has_no_issues(self):
        report = preflight_pattern_request(_request(), (0.0, 0.0, 100.0, 80.0))
        self.assertTrue(report.feasible)
        self.assertEqual(report.issues, [])

    def test_part_smaller_than_bead_is_rejected(self):
        report = preflight_pattern_request(
            _request(line_width=7.0), (0.0, 0.0, 6.0, 20.0),
        )
        self.assertEqual(report.errors[0].code, "part_smaller_than_bead")
        with self.assertRaises(InfeasibleRequestError):
            report.raise_if_infeasible()

    def test_walls_that_do_not_fit_are_rejected(self):
        report = preflight_pattern_request(
            _request(line_width=3.0, num_walls=3), (0.0, 0.0, 12.0, 30.0),
        )
        self.assertEqual(report.errors[0].code, "walls_do_not_fit")

    def test_height_below_one_layer_is_rejected(self):
        report = preflight_pattern_request(
            _request(extrude_height=0.4, layer_height=1.0),
            (0.0, 0.0, 100.0, 100.0),
        )
        self.assertEqual(report.errors[0].code, "height_below_one_layer")

    def test_height_rounding_and_density_are_warnings(self):
        report = preflight_pattern_request(
            _request(
                extrude_height=10.4,
                layer_height=1.0,
                estimated_density=32.0,
            ),
            (0.0, 0.0, 100.0, 100.0),
        )
        self.assertTrue(report.feasible)
        self.assertEqual(
            {issue.code for issue in report.warnings},
            {"height_rounded_to_layers", "density_not_reachable"},
        )

    def test_cell_pattern_degeneration_warns_but_does_not_block(self):
        report = preflight_pattern_request(
            _request(infill_spacing=8.0), (0.0, 0.0, 20.0, 50.0),
        )
        self.assertTrue(report.feasible)
        self.assertIn("pattern_degenerates", {item.code for item in report.warnings})


if __name__ == "__main__":
    unittest.main()

import unittest

from shapely.geometry import Polygon


from organic_infill_generator.infill_generator.geometry.offset_module import generate_wall_centerlines


class TestWallCenterlines(unittest.TestCase):
    def setUp(self):
        self.contour = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

    def test_outer_wall_is_compensated_by_half_line_width(self):
        walls, infill_boundary = generate_wall_centerlines(
            self.contour,
            num_walls=1,
            line_width=1.0,
            wall_spacing=0.1,
        )
        self.assertEqual(len(walls), 1)
        self.assertAlmostEqual(walls[0].area, 81.0, places=3)
        self.assertAlmostEqual(infill_boundary.area, 46.24, places=2)

    def test_num_walls_counts_total_printed_walls(self):
        walls, _ = generate_wall_centerlines(
            self.contour,
            num_walls=3,
            line_width=1.0,
            wall_spacing=0.1,
        )
        self.assertEqual(len(walls), 3)

    def test_wall_spacing_is_gap_between_surfaces(self):
        walls, _ = generate_wall_centerlines(
            self.contour,
            num_walls=2,
            line_width=1.0,
            wall_spacing=0.1,
        )
        outer_bounds = walls[0].bounds
        inner_bounds = walls[1].bounds
        self.assertAlmostEqual(inner_bounds[0] - outer_bounds[0], 1.1, places=3)
        self.assertAlmostEqual(outer_bounds[2] - inner_bounds[2], 1.1, places=3)

    def test_infill_boundary_matches_next_virtual_wall_offset(self):
        _, infill_boundary = generate_wall_centerlines(
            self.contour,
            num_walls=2,
            line_width=1.0,
            wall_spacing=0.1,
        )
        expected_bounds = (2.7, 2.7, 7.3, 7.3)
        for current, expected in zip(infill_boundary.bounds, expected_bounds):
            self.assertAlmostEqual(current, expected, places=3)

    def test_zero_walls_still_generates_infill_boundary(self):
        walls, infill_boundary = generate_wall_centerlines(
            self.contour,
            num_walls=0,
            line_width=1.0,
            wall_spacing=0.0,
        )
        self.assertEqual(len(walls), 0)
        self.assertAlmostEqual(infill_boundary.area, 81.0, places=3)


if __name__ == "__main__":
    unittest.main()

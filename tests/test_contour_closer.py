from __future__ import annotations

import unittest

import numpy as np

from organic_infill_generator.slicer import ClosureResult, ContourCloser, SimpleContourCloser


def ring_segments(points: list[tuple[float, float]]) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    return np.stack([pts, np.roll(pts, -1, axis=0)], axis=1)


SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
FAR_SQUARE = [(20.0, 0.0), (30.0, 0.0), (30.0, 10.0), (20.0, 10.0)]


class TestSimpleContourCloser(unittest.TestCase):
    def setUp(self) -> None:
        self.closer = SimpleContourCloser()

    def test_implements_the_closer_interface(self):
        self.assertIsInstance(self.closer, ContourCloser)

    def test_no_segments_gives_an_empty_result(self):
        result = self.closer.close(np.empty((0, 2, 2)))
        self.assertEqual(result.polygons, [])
        self.assertFalse(result.has_open_ends)

    def test_closes_a_ring_whatever_the_segment_order_and_direction(self):
        segments = ring_segments(SQUARE)
        generator = np.random.default_rng(7)
        shuffled = segments[generator.permutation(len(segments))]
        shuffled[::2] = shuffled[::2, ::-1]

        result = self.closer.close(shuffled)

        self.assertEqual(len(result.polygons), 1)
        self.assertAlmostEqual(result.polygons[0].area, 100.0)
        self.assertFalse(result.has_open_ends)

    def test_float_noise_across_a_cell_boundary_still_closes(self):
        # 1e-4 * k sits right on the tolerance grid, so one coordinate of the
        # pair falls just below the cell edge and the other just above it.
        edge = 0.0003
        segments = np.array([
            [[0.0, edge - 1e-12], [10.0, 0.0]],
            [[10.0, 0.0], [10.0, 10.0]],
            [[10.0, 10.0], [0.0, edge + 1e-12]],
        ])

        result = self.closer.close(segments)

        self.assertEqual(len(result.polygons), 1)
        self.assertFalse(result.has_open_ends)

    def test_a_real_gap_is_not_bridged(self):
        segments = ring_segments(SQUARE)[:-1]

        result = self.closer.close(segments)

        self.assertEqual(result.polygons, [])
        self.assertEqual(len(result.open_chains), 1)
        self.assertEqual(len(result.open_chains[0]), 4)

    def test_open_chain_is_dropped_and_the_closed_ring_is_kept(self):
        stray = np.array([[[50.0, 0.0], [60.0, 0.0]], [[60.0, 0.0], [60.0, 5.0]]])
        segments = np.concatenate([ring_segments(SQUARE), stray])

        result = self.closer.close(segments)

        self.assertEqual(len(result.polygons), 1)
        self.assertEqual(len(result.open_chains), 1)
        self.assertTrue(result.has_open_ends)

    def test_tail_attached_to_a_ring_does_not_lose_the_ring(self):
        tail = np.array([[[10.0, 10.0], [15.0, 15.0]], [[15.0, 15.0], [18.0, 15.0]]])
        segments = np.concatenate([ring_segments(SQUARE), tail])

        result = self.closer.close(segments)

        self.assertEqual(len(result.polygons), 1)
        self.assertAlmostEqual(result.polygons[0].area, 100.0)
        self.assertEqual(len(result.open_chains), 1)

    def test_two_rings_stay_separate(self):
        segments = np.concatenate([ring_segments(SQUARE), ring_segments(FAR_SQUARE)])

        result = self.closer.close(segments)

        self.assertEqual(sorted(p.area for p in result.polygons), [100.0, 100.0])

    def test_two_rings_touching_at_one_point_come_out_as_two_rings(self):
        other = [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)]
        segments = np.concatenate([ring_segments(SQUARE), ring_segments(other)])

        result = self.closer.close(segments)

        self.assertEqual(sorted(round(p.area) for p in result.polygons), [100, 100])
        self.assertFalse(result.has_open_ends)

    def test_repeated_and_collapsed_segments_are_ignored(self):
        segments = ring_segments(SQUARE)
        extra = np.array([segments[0], segments[1, ::-1], [[3.0, 3.0], [3.0, 3.0]]])

        result = self.closer.close(np.concatenate([segments, extra]))

        self.assertEqual(len(result.polygons), 1)
        self.assertFalse(result.has_open_ends)

    def test_exact_coordinates_are_kept(self):
        points = [(0.123456789, 0.0), (10.0, 0.0), (10.0, 10.0)]

        polygon = self.closer.close(ring_segments(points)).polygons[0]

        self.assertIn(points[0], list(polygon.exterior.coords))

    def test_rings_do_not_depend_on_the_segment_order(self):
        segments = np.concatenate([ring_segments(SQUARE), ring_segments(FAR_SQUARE)])

        first = self.closer.close(segments)
        second = self.closer.close(segments[::-1])

        self.assertEqual(
            sorted(sorted(set(p.exterior.coords)) for p in first.polygons),
            sorted(sorted(set(p.exterior.coords)) for p in second.polygons),
        )

    def test_invalid_tolerance_is_rejected(self):
        with self.assertRaises(ValueError):
            SimpleContourCloser(tolerance=0.0).close(ring_segments(SQUARE))


class TestCloserInterface(unittest.TestCase):
    def test_any_object_with_close_is_a_contour_closer(self):
        class Recording:
            def close(self, segments: np.ndarray, model_diag: float) -> ClosureResult:
                return ClosureResult()

        self.assertIsInstance(Recording(), ContourCloser)


if __name__ == "__main__":
    unittest.main()

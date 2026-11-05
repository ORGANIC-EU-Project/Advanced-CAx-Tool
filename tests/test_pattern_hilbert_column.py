import unittest

import numpy as np


from organic_infill_generator.infill_generator.patterns.pattern_hilbert_column import HilbertColumnPattern


class TestHilbertColumnPattern(unittest.TestCase):
    def test_hilbert_column_spans_bottom_to_top_across_levels(self) -> None:
        pattern = HilbertColumnPattern()

        for pattern_step in (1.0, 0.5, 0.25, 0.12, 0.08):
            with self.subTest(pattern_step=pattern_step):
                points = pattern.generate(
                    x_center=0.0,
                    y_start=0.0,
                    y_end=10.0,
                    pattern_amplitude=2.0,
                    pattern_step=pattern_step,
                    smoothing_passes=0,
                )

                self.assertGreaterEqual(points.shape[0], 2)
                self.assertTrue(np.isclose(points[0, 1], 0.0))
                self.assertTrue(np.isclose(points[-1, 1], 10.0))

    def test_smoothing_preserves_opposite_vertical_endpoints(self) -> None:
        pattern = HilbertColumnPattern()

        points = pattern.generate(
            x_center=1.0,
            y_start=-2.0,
            y_end=8.0,
            pattern_amplitude=3.0,
            pattern_step=0.25,
            smoothing_passes=2,
        )

        self.assertGreaterEqual(points.shape[0], 2)
        self.assertTrue(np.isclose(points[0, 1], -2.0))
        self.assertTrue(np.isclose(points[-1, 1], 8.0))


if __name__ == "__main__":
    unittest.main()

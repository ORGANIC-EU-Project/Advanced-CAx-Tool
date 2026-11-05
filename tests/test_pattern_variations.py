import unittest


from organic_infill_generator.infill_generator.models import PatternRequest, PatternVariationRequest
from organic_infill_generator.infill_generator.patterns.pattern_variations import generate_density_variations


class TestPatternVariations(unittest.TestCase):
    @staticmethod
    def _build_request(infill_density: float) -> PatternRequest:
        return PatternRequest(
            num_walls=2,
            line_width=0.4,
            wall_spacing=0.0,
            fillet_radius=0.4,
            infill_density=infill_density,
            infill_pattern="honeycomb",
            extrude_height=0.0,
        )

    def test_two_variations_are_balanced_around_original_density(self):
        request = self._build_request(50.0)

        variations = generate_density_variations(
            PatternVariationRequest(
                original_pattern=request,
                n_variations=2,
                variation_range=20.0,
                step=5.0,
            ),
        )

        self.assertEqual([item.infill_density for item in variations], [45.0, 55.0])

    def test_odd_count_stays_balanced_and_expands_by_magnitude(self):
        request = self._build_request(50.0)

        variations = generate_density_variations(
            PatternVariationRequest(
                original_pattern=request,
                n_variations=3,
                variation_range=20.0,
                step=5.0,
            ),
        )

        self.assertEqual([item.infill_density for item in variations], [45.0, 55.0, 40.0])

    def test_values_are_clamped_to_logical_density_bounds(self):
        request = self._build_request(10.0)

        variations = generate_density_variations(
            PatternVariationRequest(
                original_pattern=request,
                n_variations=4,
                variation_range=20.0,
                step=5.0,
            ),
        )

        self.assertEqual([item.infill_density for item in variations], [5.0, 15.0, 0.0, 20.0])

    def test_variations_keep_other_request_fields(self):
        request = self._build_request(50.0)

        variation = generate_density_variations(
            PatternVariationRequest(
                original_pattern=request,
                n_variations=1,
                variation_range=10.0,
                step=5.0,
            )
        )[0]

        self.assertEqual(variation.num_walls, request.num_walls)
        self.assertEqual(variation.line_width, request.line_width)
        self.assertEqual(variation.infill_pattern, request.infill_pattern)
        self.assertNotEqual(variation.infill_density, request.infill_density)

    def test_raises_when_more_unique_variations_are_requested_than_possible(self):
        request = self._build_request(50.0)

        with self.assertRaises(ValueError):
            generate_density_variations(
                PatternVariationRequest(
                    original_pattern=request,
                    n_variations=5,
                    variation_range=10.0,
                    step=5.0,
                )
            )


if __name__ == "__main__":
    unittest.main()

"""Toolpaths must not change unnoticed. See regenerate_golden.py to update them."""
from __future__ import annotations

import json
import unittest

from regenerate_golden import GOLDEN, measure


class TestGoldenToolpaths(unittest.TestCase):
    def test_every_pattern_matches_its_golden_record(self):
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        self.assertEqual(len(golden), 10)
        for pattern, expected in golden.items():
            with self.subTest(pattern=pattern):
                actual = measure(pattern)
                self.assertEqual(actual["points"], expected["points"])
                self.assertAlmostEqual(actual["length_mm"], expected["length_mm"], places=2)
                self.assertEqual(actual["start"], expected["start"])
                self.assertEqual(actual["end"], expected["end"])


if __name__ == "__main__":
    unittest.main()

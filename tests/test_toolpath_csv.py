from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import LineString

from organic_infill_generator.infill_generator.generator import (
    TOOLPATH_CSV_COLUMNS,
    save_toolpath_csv,
)
from organic_infill_generator.infill_generator.settings import PathSettings


class _Preview:
    """Minimal preview shape consumed by save_toolpath_csv()."""

    def __init__(
            self,
            points: list[list[float]],
            *,
            z: float,
            title: str = "robot path",
            rounded_points: list[list[float]] | None = None,
    ) -> None:
        self.path_points = np.asarray(points, dtype=float)
        self.mesh_preview_geometry = (
            LineString(rounded_points) if rounded_points is not None else None
        )
        self.mesh_kind = "infill"
        self.fillet_radius = 1.0 if rounded_points is not None else 0.0
        self.line_width = 1.0
        self.layer_height = 2.0
        self.z = z
        self.title = title


class TestToolpathCsv(unittest.TestCase):
    def setUp(self) -> None:
        self._previous_results_dir = PathSettings.RESULTS_DIR
        self._temporary_directory = tempfile.TemporaryDirectory()
        PathSettings.RESULTS_DIR = self._temporary_directory.name

    def tearDown(self) -> None:
        PathSettings.RESULTS_DIR = self._previous_results_dir
        self._temporary_directory.cleanup()

    def test_writes_tagged_schema_and_orders_layers_by_z(self) -> None:
        previews = [
            _Preview([[20.0, 21.0], [22.0, 23.0]], z=3.0),
            _Preview([[10.0, 11.0], [12.0, 13.0]], z=1.0),
        ]

        output_path = Path(save_toolpath_csv(previews))

        with output_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)

        self.assertEqual(reader.fieldnames, list(TOOLPATH_CSV_COLUMNS))
        self.assertEqual([row["layer"] for row in rows], ["1", "1", "2", "2"])
        self.assertEqual(
            [row["x"] for row in rows],
            ["10.000000", "12.000000", "20.000000", "22.000000"],
        )
        self.assertEqual(
            [row["z"] for row in rows],
            ["1.000000", "1.000000", "3.000000", "3.000000"],
        )
        self.assertEqual([row["path_id"] for row in rows], [
            "L001_flow_path",
            "L001_flow_path",
            "L002_flow_path",
            "L002_flow_path",
        ])
        self.assertEqual([row["point_index"] for row in rows], ["0", "1", "0", "1"])
        self.assertEqual({row["print_type"] for row in rows}, {"resin"})
        self.assertEqual({row["feature_type"] for row in rows}, {"flow"})
        self.assertEqual({row["path_type"] for row in rows}, {"infill"})

    def test_uses_the_rounded_path_for_filleted_layers(self) -> None:
        preview = _Preview(
            [[0.0, 0.0], [2.0, 0.0]],
            z=1.0,
            rounded_points=[[0.0, 0.0], [1.0, 0.5], [2.0, 0.0]],
        )

        output_path = Path(save_toolpath_csv([preview]))

        with output_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual(
            [(row["x"], row["y"]) for row in rows],
            [
                ("0.000000", "0.000000"),
                ("1.000000", "0.500000"),
                ("2.000000", "0.000000"),
            ],
        )

    def test_rejects_an_empty_stack(self) -> None:
        with self.assertRaisesRegex(ValueError, "No layer previews"):
            save_toolpath_csv([])


if __name__ == "__main__":
    unittest.main()

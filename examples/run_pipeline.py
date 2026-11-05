"""
The complete process of the tool on one part.

    python examples/run_pipeline.py                     # a generated demo panel
    python examples/run_pipeline.py path/to/part.stl    # your own STEP or STL

Outputs go to results/demo/: robot toolpath CSV, STL, contours and a summary.
A STEP input needs the optional CAD backend: pip install -e .[cad]
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import trimesh

from organic_infill_generator import PipelineConfig, run_pipeline

# Process parameters, in mm and percent.
CONFIG = PipelineConfig(
    layer_height=6.0,
    line_width=6.0,
    infill_density=40.0,
    infill_pattern="honeycomb",
    num_walls=1,
    wall_spacing=0.2,
    fillet_radius=6.0,
    title="demo",
    output_dir="results/demo",
)


def rounded_ring(width: float, depth: float, radius: float, z: float, arc_points: int = 8) -> np.ndarray:
    """A rounded rectangle centred on the origin, counter-clockwise, at height z."""
    corners = ((1, 1, 0.0), (-1, 1, 90.0), (-1, -1, 180.0), (1, -1, 270.0))
    points = []
    for sign_x, sign_y, start in corners:
        center = np.array([sign_x * (width / 2.0 - radius), sign_y * (depth / 2.0 - radius)])
        for angle in np.radians(np.linspace(start, start + 90.0, arc_points)):
            points.append([*(center + radius * np.array([np.cos(angle), np.sin(angle)])), z])
    return np.asarray(points)


def write_demo_part(path: Path) -> Path:
    """A closed drafted panel: its section shrinks with height, so every layer differs."""
    bottom = rounded_ring(600.0, 400.0, 60.0, 0.0)
    top = rounded_ring(420.0, 300.0, 30.0, 240.0)
    count = len(bottom)
    vertices = np.vstack([bottom, top, [[0.0, 0.0, 0.0]], [[0.0, 0.0, 240.0]]])
    bottom_center, top_center = 2 * count, 2 * count + 1
    faces = []
    for index in range(count):
        following = (index + 1) % count
        faces.append([index, following, following + count])
        faces.append([index, following + count, index + count])
        faces.append([bottom_center, following, index])
        faces.append([top_center, index + count, following + count])
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=vertices, faces=faces, process=True).export(path)
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    if len(sys.argv) > 1:
        part = Path(sys.argv[1])
    else:
        part = write_demo_part(Path(CONFIG.output_dir) / "demo_part.stl")

    result = run_pipeline(part, CONFIG)

    slicer = result.summary["slicer"]
    validation = result.summary["validation"]
    print(f"Layers closed : {slicer['closed_layers']} of {slicer['total_layers']}")
    print(f"Toolpaths     : {validation['layers_with_toolpath']} layers, {validation['total_jumps']} jumps")
    print(f"Path length   : {validation['total_path_length_mm'] / 1000.0:.1f} m")
    print(f"Robot CSV     : {result.toolpath_csv_path}")
    print(f"STL           : {result.output_stl_path}")
    print(f"Summary       : {result.summary_path}")


if __name__ == "__main__":
    main()

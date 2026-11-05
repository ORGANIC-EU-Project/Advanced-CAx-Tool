"""Small parts and requests shared by the tests. Built in code, so no binary is stored."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


def write_box_stl(path: Path, extents: tuple[float, float, float] = (120.0, 80.0, 12.0)) -> Path:
    """A closed prism with its corner at the origin."""
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(np.asarray(extents) / 2.0)
    mesh.export(path)
    return path


def drafted_mesh(
        bottom: float = 150.0,
        shrink: tuple[float, float] = (40.0, 18.0),
        height: float = 36.0,
) -> trimesh.Trimesh:
    """A frustum whose contour shrinks by a different amount on each side."""
    left, right = shrink
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [bottom, 0.0, 0.0],
        [bottom, 100.0, 0.0],
        [0.0, 100.0, 0.0],
        [left, 10.0, height],
        [bottom - right, 10.0, height],
        [bottom - right, 90.0, height],
        [left, 90.0, height],
    ])
    faces = np.array([
        [0, 2, 1], [0, 3, 2],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=True)


def write_drafted_stl(path: Path) -> Path:
    drafted_mesh().export(path)
    return path


def write_tube_stl(path: Path, outer: float = 50.0, inner: float = 25.0, height: float = 10.0) -> Path:
    """A round tube: every layer is a contour with a hole."""
    mesh = trimesh.creation.annulus(r_min=inner, r_max=outer, height=height, sections=64)
    mesh.apply_translation(-mesh.bounds[0])
    mesh.export(path)
    return path


def layer(z: float, exterior: list[list[float]]) -> dict:
    return {
        "z": z,
        "method": "exact",
        "contours": [{
            "id": 0,
            "area": 1.0,
            "is_largest": True,
            "exterior": exterior,
            "holes": [],
        }],
        "open_lines": [],
    }


def slicer_output(layers: list[dict], layer_height: float = 1.5) -> dict:
    """A SlicerOutput-like dict whose bounds enclose ``layers``."""
    points = np.vstack([np.asarray(item["contours"][0]["exterior"], dtype=float) for item in layers])
    low, high = points.min(axis=0), points.max(axis=0)
    height = len(layers) * layer_height
    return {
        "layer_height": layer_height,
        "model_bounds": {
            "min": [float(low[0]), float(low[1]), 0.0],
            "max": [float(high[0]), float(high[1]), height],
            "size": [float(high[0] - low[0]), float(high[1] - low[1]), height],
        },
        "layers": layers,
        "total_layers": len(layers),
        "closed_layers": len(layers),
        "failed_layers": 0,
        "noise_layers": 0,
    }


def request(**overrides) -> dict:
    """A printable large-format request: 6 mm bead, 1.5 mm layer."""
    values = {
        "num_walls": 1,
        "line_width": 6.0,
        "wall_spacing": 0.2,
        "fillet_radius": 6.0,
        "infill_density": 40.0,
        "infill_pattern": "honeycomb",
        "extrude_height": 1.5,
        "layer_height": 1.5,
        "bottom_layers": 0,
        "top_layers": 0,
    }
    values.update(overrides)
    return values

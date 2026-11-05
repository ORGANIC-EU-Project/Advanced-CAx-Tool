"""Core STL slicing pipeline."""
from __future__ import annotations

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import trimesh

from .closure import ContourCloser, SimpleContourCloser
from .config import NOISE_SEGMENT_THRESHOLD, logger
from .geometry_utils import (
    analyze_model_scale,
    clean_polys_from_list,
    organize_polygons_hierarchy,
)
from .models import BoundingBox, ContourPolygon, SlicerLayer, SlicerOutput, SlicerResult


class SlicerStats:
    """Accumulate layer outcomes while slicing an STL."""

    def __init__(self):
        self.total_layers = 0
        self.exact_layers = 0     # every segment joined a closed contour
        self.partial_layers = 0   # closed contours, plus open chains that were dropped
        self.failed_layers = 0    # open chains and no closed contour
        self.noise_layers = 0     # too few segments to be a real contour
        self.layer_details = []

    def add_layer(self, z: float, method: str, segment_count: int, polygon_count: int):
        """Record one layer and classify it as exact, partial, failed, or noise."""
        self.total_layers += 1
        is_noise = segment_count < NOISE_SEGMENT_THRESHOLD
        if is_noise:
            self.noise_layers += 1
        elif method == 'failed':
            self.failed_layers += 1
        elif method == 'partial':
            self.partial_layers += 1
        else:
            self.exact_layers += 1
        self.layer_details.append({
            'z': float(z),
            'method': 'noise' if is_noise else method,
            'segments': segment_count,
            'polygons': polygon_count,
        })

    def report(self) -> str:
        """Return a compact one-line summary suitable for logs."""
        closed = self.exact_layers + self.partial_layers
        return (
            f"Total: {self.total_layers} | Closed: {closed} "
            f"(exact={self.exact_layers}, partial={self.partial_layers}) | "
            f"Failed: {self.failed_layers} | Noise: {self.noise_layers}"
        )


def get_raw_intersection(mesh: trimesh.Trimesh, z: float) -> np.ndarray:
    """Intersect the mesh with a horizontal plane and return XY segments."""
    try:
        lines_3d = trimesh.intersections.mesh_plane(
            mesh, plane_normal=[0, 0, 1], plane_origin=[0, 0, z])
        if len(lines_3d) == 0:
            return np.empty((0, 2, 2))
        return lines_3d[:, :, :2]
    except Exception as e:
        logger.debug(f"Intersection error at z={z}: {e}")
        return np.empty((0, 2, 2))


def process_single_layer(
    mesh: trimesh.Trimesh,
    z: float,
    model_diag: float,
    contour_closer: ContourCloser,
) -> Tuple[float, Dict[str, Any]]:
    """
    Close the contours of a single layer.
    Returns (z, layer_data).
    """
    raw_seg = get_raw_intersection(mesh, z)
    closure = contour_closer.close(raw_seg, model_diag)
    clean_polys, skipped = clean_polys_from_list(closure.polygons)

    if not closure.has_open_ends:
        method = 'exact'
    elif clean_polys:
        # The chains that did not close are left out of the layer: the closed
        # contours are what can be printed.
        method = 'partial'
    else:
        method = 'failed'

    layer_data = {
        'raw_segments': raw_seg,
        'polygons': organize_polygons_hierarchy(clean_polys),
        'skipped': skipped,
        'open_lines': closure.open_chains,
        'method': method,
        'sample_z': z,
    }
    return z, layer_data


def compute_slices(
    mesh: trimesh.Trimesh,
    layer_height: float,
    stats: SlicerStats,
    contour_closer: ContourCloser,
    parallel: bool = True,
    num_workers: int = None,
) -> Dict[float, Dict]:
    """Slice all Z levels and close each layer into contour polygons."""
    z_max = mesh.bounds[1, 2]
    model_diag = float(np.linalg.norm(mesh.extents))
    num_layers = int(np.ceil(z_max / layer_height))
    # Each layer is cut at its mid-plane. A last partial layer whose mid-plane
    # falls above the part has nothing to cut.
    z_levels = [z for z in (i * layer_height + layer_height / 2 for i in range(num_layers))
                if z < z_max]

    logger.info(f"Slicing: {num_layers} layers @ {layer_height}mm | "
                f"model_diag={model_diag:.1f}mm | "
                f"{'PARALLEL' if parallel else 'SEQUENTIAL'}")

    layers_data: Dict[float, Dict] = {}

    if parallel and len(z_levels) > 1:
        nw = num_workers or max(1, multiprocessing.cpu_count() - 1)
        logger.info(f"Using {nw} worker processes")
        # Each worker receives the same mesh snapshot and independently closes
        # one Z level. Results are gathered by their Z value so output ordering
        # remains deterministic later in _layers_to_output().
        with ProcessPoolExecutor(max_workers=nw) as executor:
            futures = {
                executor.submit(process_single_layer, mesh, z, model_diag, contour_closer): z
                for z in z_levels
            }
            for future in as_completed(futures):
                z, layer_data = future.result()
                layers_data[z] = layer_data
    else:
        for z in z_levels:
            z, layer_data = process_single_layer(mesh, z, model_diag, contour_closer)
            layers_data[z] = layer_data

    for z in sorted(layers_data):
        layer_data = layers_data[z]
        stats.add_layer(z, layer_data['method'],
                        len(layer_data['raw_segments']),
                        len(layer_data['polygons']))

    logger.info(stats.report())
    return layers_data


def _load_and_prep_mesh(input_file: str) -> trimesh.Trimesh:
    """Load STL, autoscale if needed, center XY at origin, move Z floor to 0."""
    mesh = trimesh.load(str(input_file), process=False)
    # An STL repeats every vertex per triangle, so closure is only visible once
    # they are merged. The slicer itself keeps the file's coordinates untouched.
    merged = mesh.copy()
    merged.merge_vertices()
    if not merged.is_watertight:
        logger.warning("Mesh is NOT watertight: expect incomplete contours on some layers")
    else:
        logger.info("Mesh is watertight")

    scale_info = analyze_model_scale(mesh)
    logger.info(f"Scale: {scale_info['scale_name']} | {scale_info['reason']}")
    if scale_info['scale_factor'] != 1.0:
        mesh.apply_scale(scale_info['scale_factor'])

    mesh.apply_translation(-mesh.bounds[0])
    logger.info(f"Bounds after aligning to origin: {mesh.bounds}")
    return mesh


def _layers_to_output(
    layers_data: Dict[float, Dict],
    mesh_bounds: np.ndarray,
    layer_height: float,
    stats: SlicerStats,
) -> SlicerOutput:
    """Convert internal layers_data dict to the public SlicerOutput dataclass."""
    slicer_layers: List[SlicerLayer] = []

    for z in sorted(layers_data.keys()):
        data = layers_data[z]
        polys = data['polygons']

        largest_idx = (
            max(range(len(polys)), key=lambda i: polys[i].area) if polys else -1
        )

        contours = [
            ContourPolygon(
                id=i,
                area=float(poly.area),
                is_largest=(i == largest_idx),
                exterior=[[float(x), float(y)] for x, y in poly.exterior.coords[:-1]],
                holes=[
                    [[float(x), float(y)] for x, y in ring.coords[:-1]]
                    for ring in poly.interiors
                ],
            )
            for i, poly in enumerate(polys)
        ]

        open_lines = [
            [[float(x), float(y)] for x, y in chain]
            for chain in data.get('open_lines', [])
        ]

        slicer_layers.append(SlicerLayer(
            z=float(z),
            method=data['method'],
            contours=contours,
            open_lines=open_lines,
        ))

    model_bounds = BoundingBox(
        min=[float(v) for v in mesh_bounds[0]],
        max=[float(v) for v in mesh_bounds[1]],
        size=[float(v) for v in (mesh_bounds[1] - mesh_bounds[0])],
    )

    closed = stats.exact_layers + stats.partial_layers
    return SlicerOutput(
        layer_height=layer_height,
        model_bounds=model_bounds,
        layers=slicer_layers,
        total_layers=stats.total_layers,
        closed_layers=closed,
        failed_layers=stats.failed_layers,
        noise_layers=stats.noise_layers,
    )


def slice_contours(
    input_file: str,
    layer_height: float = 0.4,
    contour_closer: ContourCloser | None = None,
    parallel: bool = True,
    num_workers: int = None,
) -> SlicerResult:
    """
    Slice an STL and return a SlicerResult.

    The model is auto-centered (XY at origin, Z floor at 0) before slicing.
    ``contour_closer`` decides how each layer's segments become closed contours;
    the default is :class:`SimpleContourCloser`.
    Pass the result to save_contours() and/or show_contours() as needed.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"STL not found: {input_file}")

    mesh = _load_and_prep_mesh(input_file)
    stats = SlicerStats()
    layers_data = compute_slices(
        mesh, layer_height, stats,
        contour_closer=contour_closer if contour_closer is not None else SimpleContourCloser(),
        parallel=parallel,
        num_workers=num_workers,
    )
    output = _layers_to_output(layers_data, mesh.bounds, layer_height, stats)
    return SlicerResult(output=output, layers_data=layers_data, mesh_bounds=mesh.bounds)


def save_contours(result: SlicerResult, path: str) -> None:
    """Serialize a SlicerResult's output to JSON."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(asdict(result.output), f, indent=2)
    logger.info(f"Contours saved: {path}")

"""Serializable slicer data models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class ContourPolygon:
    """One closed contour (shell + optional holes) within a layer."""
    id: int
    area: float                             # mm2
    is_largest: bool                        # largest polygon in this layer
    exterior: List[List[float]]             # [[x, y], ...]  (closing point excluded)
    holes: List[List[List[float]]] = field(default_factory=list)  # [[[x, y], ...], ...]


@dataclass
class SlicerLayer:
    """All contours at a given Z height."""
    z: float
    method: str                             # 'exact' | 'partial' | 'failed'
    contours: List[ContourPolygon] = field(default_factory=list)
    open_lines: List[List[List[float]]] = field(default_factory=list)  # unclosed chains


@dataclass
class BoundingBox:
    """Model bounding box after centering (XY at origin, Z starts at 0)."""
    min: List[float]    # [x_min, y_min, z_min]
    max: List[float]    # [x_max, y_max, z_max]
    size: List[float]   # [width, depth, height]


@dataclass
class SlicerOutput:
    """Complete result of one slicing run."""
    layer_height: float
    model_bounds: BoundingBox
    layers: List[SlicerLayer]
    total_layers: int
    closed_layers: int   # exact + partial
    failed_layers: int   # open chains and no closed contour
    noise_layers: int    # too few segments: tangential grazes


class SlicerResult:
    """
    Returned by slice_contours(). Holds everything from one slicing run:
    - output      : SlicerOutput, serializable contour data (pass to save_contours)
    - _layers_data: internal dict with raw geometry for visualization (pass to show_contours)
    - _mesh_bounds: np.ndarray [2, 3] for the viewer axes
    """

    def __init__(
        self,
        output: SlicerOutput,
        layers_data: Dict[float, Dict],
        mesh_bounds: np.ndarray,
    ):
        self.output = output
        self._layers_data = layers_data
        self._mesh_bounds = mesh_bounds

"""Geometry utilities used by the slicer pipeline."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import trimesh
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import make_valid

from .config import (
    LARGE_SCALE_THRESHOLD,
    MIN_AREA_THRESHOLD,
    SMALL_SCALE_THRESHOLD,
    SUB_MM_AUTOSCALE_FACTOR,
)

class HierarchyNode:
    """Small tree node used to recover shell/hole nesting from flat polygons."""

    def __init__(self, polygon):
        self.polygon = polygon
        self.children = []

    def add_child(self, new_node):
        """Insert a contained polygon at the deepest matching level."""
        for child in self.children:
            if child.polygon.contains(new_node.polygon):
                child.add_child(new_node)
                return
        self.children.append(new_node)


def organize_polygons_hierarchy(flat_polygons: List[Polygon]) -> List[Polygon]:
    """
    Convert flat slice polygons into solids with holes.

    Trimesh/shapely can return every contour as a standalone polygon. This
    rebuilds containment: outer polygons become shells, their direct children
    become holes, and grandchildren become new solid islands.
    """
    if not flat_polygons:
        return []
    sorted_polys = sorted(flat_polygons, key=lambda p: p.area, reverse=True)
    nodes = [HierarchyNode(p) for p in sorted_polys]
    roots = []
    for node in nodes:
        placed = False
        for root in roots:
            if root.polygon.contains(node.polygon):
                root.add_child(node)
                placed = True
                break
        if not placed:
            roots.append(node)

    final = []

    def traverse(node, is_solid):
        if is_solid:
            shell = node.polygon.exterior
            holes = [c.polygon.exterior for c in node.children]
            final.append(Polygon(shell=shell, holes=holes))
            for child in node.children:
                for grandchild in child.children:
                    traverse(grandchild, True)

    for root in roots:
        traverse(root, True)
    return final


def format_extents(extents: np.ndarray) -> str:
    """Format XYZ extents for scale diagnostics."""
    return f"X={extents[0]:.6g}, Y={extents[1]:.6g}, Z={extents[2]:.6g}"


def analyze_model_scale(mesh: trimesh.Trimesh) -> Dict[str, Any]:
    """Classify the mesh unit scale and return an optional autoscale factor."""
    extents = mesh.extents.copy()
    max_extent = float(np.max(np.abs(extents))) if len(extents) else 0.0
    if max_extent > LARGE_SCALE_THRESHOLD:
        return {'scale_name': 'meter-scale', 'scale_factor': 1.0,
                'reason': 'Max extent above 1000.', 'extents': extents}
    if 0 < max_extent < SMALL_SCALE_THRESHOLD:
        return {'scale_name': 'sub-mm', 'scale_factor': SUB_MM_AUTOSCALE_FACTOR,
                'reason': 'Max extent below 0.01. Rescaled x1000.', 'extents': extents}
    return {'scale_name': 'mm-scale', 'scale_factor': 1.0,
            'reason': 'Model stays at current scale.', 'extents': extents}


def clean_polys_from_list(raw_polys: List[Polygon]) -> Tuple[List[Polygon], List[Polygon]]:
    """make_valid + area filter. Returns (clean, skipped). No fillet applied here."""
    clean, skipped = [], []
    for p in raw_polys:
        if p.area < MIN_AREA_THRESHOLD:
            skipped.append(p)
            continue
        p = make_valid(p)
        if isinstance(p, Polygon) and p.is_valid and p.area > MIN_AREA_THRESHOLD:
            clean.append(p)
        elif isinstance(p, MultiPolygon):
            clean.extend(g for g in p.geoms
                         if g.is_valid and g.area > MIN_AREA_THRESHOLD)
        else:
            skipped.append(p)
    return clean, skipped

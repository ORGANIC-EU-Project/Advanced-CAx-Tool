"""
Shared plot utilities for pipeline previews, self-tests, and visual tests.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import numpy as np
from shapely.geometry import LineString, Point, Polygon, base

from ..mesh.extrusion_module import _build_swept_geometry
from ..settings import ApiDefaults, FilletSettings, GeometrySettings, PathSettings, PlotSettings


def should_show_plots() -> bool:
    """Return true when test/demo plots should be shown interactively."""
    return os.environ.get("SHOW_TEST_PLOTS", "").strip().lower() in {"1", "true", "yes", "on"}


def should_save_plots() -> bool:
    """Return true when test/demo plots should be written to results."""
    return os.environ.get("SAVE_TEST_PLOTS", "").strip().lower() in {"1", "true", "yes", "on"}


def should_force_agg() -> bool:
    """Return true when matplotlib should prefer a non-interactive backend."""
    return not should_show_plots()


def finalize_figure(fig, filename: str | None = None) -> str | None:
    """
    Save and/or show a matplotlib figure depending on environment flags.
    """
    output_path = None

    if should_save_plots() and filename:
        os.makedirs(PathSettings.RESULTS_DIR, exist_ok=True)
        output_path = os.path.join(PathSettings.RESULTS_DIR, filename)
        fig.savefig(output_path, dpi=PlotSettings.SAVE_DPI)

    if should_show_plots() and "agg" not in plt.get_backend().lower():
        plt.show()

    plt.close(fig)
    return output_path


def show_figure(fig) -> None:
    """
    Fast path for module self-tests: just show the figure and close it.
    """
    plt.show()
    plt.close(fig)


def plot_geometry(
        ax: plt.Axes,
        geom: base.BaseGeometry,
        edge_only: bool = False,
        label: str | None = None,
        linewidth: float = PlotSettings.DEFAULT_LINEWIDTH,
        **kwargs,
):
    """
    Plot any Shapely geometry onto a matplotlib Axes.

    Handles Polygon, MultiPolygon, LineString, MultiLineString,
    Point, and GeometryCollection recursively.
    """
    if geom.is_empty:
        return

    def _signed_ring_area(coords: np.ndarray) -> float:
        if coords.shape[0] < 3:
            return 0.0
        x = coords[:, 0]
        y = coords[:, 1]
        return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))

    def _ring_to_vertices_codes(coords: np.ndarray) -> tuple[list[tuple[float, float]], list[int]]:
        if coords.shape[0] == 0:
            return [], []
        if coords.shape[0] >= 2 and np.allclose(coords[0], coords[-1]):
            coords = coords[:-1]
        if coords.shape[0] == 0:
            return [], []
        vertices = [tuple(map(float, coords[0]))]
        codes = [Path.MOVETO]
        for point in coords[1:]:
            vertices.append(tuple(map(float, point)))
            codes.append(Path.LINETO)
        vertices.append(tuple(map(float, coords[0])))
        codes.append(Path.CLOSEPOLY)
        return vertices, codes

    def _polygon_to_path(poly: Polygon) -> Path:
        vertices: list[tuple[float, float]] = []
        codes: list[int] = []

        exterior = np.asarray(poly.exterior.coords, dtype=float)
        exterior_sign = np.sign(_signed_ring_area(exterior))
        ext_vertices, ext_codes = _ring_to_vertices_codes(exterior)
        vertices.extend(ext_vertices)
        codes.extend(ext_codes)

        for interior in poly.interiors:
            interior_coords = np.asarray(interior.coords, dtype=float)
            interior_sign = np.sign(_signed_ring_area(interior_coords))
            if exterior_sign != 0 and interior_sign == exterior_sign:
                interior_coords = interior_coords[::-1]
            hole_vertices, hole_codes = _ring_to_vertices_codes(interior_coords)
            vertices.extend(hole_vertices)
            codes.extend(hole_codes)

        return Path(np.asarray(vertices, dtype=float), codes)

    def _plot_polygon(poly: Polygon, polygon_label: str | None = None):
        x, y = poly.exterior.xy
        if edge_only:
            ax.plot(x, y, linewidth=linewidth, label=polygon_label, **kwargs)
        else:
            fill_kwargs = dict(kwargs)
            fill_kwargs.setdefault("alpha", PlotSettings.FILL_ALPHA)
            facecolor = fill_kwargs.pop("color", None)
            if facecolor is not None:
                fill_kwargs.setdefault("facecolor", facecolor)
                fill_kwargs.setdefault("edgecolor", facecolor)
            fill_kwargs.setdefault("linewidth", linewidth)
            if polygon_label is not None:
                fill_kwargs.setdefault("label", polygon_label)
            patch = PathPatch(_polygon_to_path(poly), **fill_kwargs)
            if hasattr(patch, "set_fillrule"):
                patch.set_fillrule("evenodd")
            ax.add_patch(patch)
            return
        for interior in poly.interiors:
            ix, iy = interior.xy
            ax.plot(ix, iy, linewidth=linewidth, **kwargs)

    gtype = geom.geom_type

    if gtype == "Polygon":
        _plot_polygon(geom, label)
    elif gtype == "MultiPolygon":
        first = True
        for poly in geom.geoms:
            _plot_polygon(poly, label if first else None)
            first = False
    elif gtype in ("LineString", "LinearRing"):
        x, y = geom.xy
        ax.plot(x, y, linewidth=linewidth, label=label, **kwargs)
    elif gtype == "MultiLineString":
        first = True
        for line in geom.geoms:
            x, y = line.xy
            ax.plot(x, y, linewidth=linewidth, label=label if first else None, **kwargs)
            first = False
    elif gtype == "Point":
        ax.scatter([geom.x], [geom.y], s=20, label=label, **kwargs)
    elif gtype == "GeometryCollection":
        first = True
        for subgeom in geom.geoms:
            plot_geometry(
                ax,
                subgeom,
                edge_only=edge_only,
                label=label if first else None,
                linewidth=linewidth,
                **kwargs,
            )
            first = False
    else:
        try:
            x, y = geom.xy
            ax.plot(x, y, linewidth=linewidth, label=label, **kwargs)
        except Exception:
            pass


def plot_path_points(
        ax: plt.Axes,
        points: np.ndarray,
        *,
        color: str = "royalblue",
        linewidth: float = 1.0,
        label: str | None = None,
        alpha: float = 1.0,
        zorder: float | None = None,
) -> None:
    """
    Plot a 2D point sequence as a line using the shared geometry renderer.
    """
    coords = np.asarray(points, dtype=float)
    if coords.ndim != 2 or coords.shape[0] < 2 or coords.shape[1] != 2:
        return

    kwargs: dict[str, object] = {
        "color": color,
        "alpha": alpha,
    }
    if zorder is not None:
        kwargs["zorder"] = zorder
    plot_geometry(
        ax,
        LineString(coords),
        edge_only=True,
        linewidth=linewidth,
        label=label,
        **kwargs,
    )


def _interpolate_color(start_color: str, end_color: str, t: float) -> tuple[float, float, float]:
    start = np.asarray(mcolors.to_rgb(start_color), dtype=float)
    end = np.asarray(mcolors.to_rgb(end_color), dtype=float)
    return tuple((1.0 - t) * start + t * end)


def _build_plot_linewidth_geometries(preview_result, line_width: float) -> list[base.BaseGeometry]:
    """Build per-segment ribbon polygons for the line-width preview overlay."""
    if line_width <= 0:
        return []

    source_geometry = preview_result.mesh_preview_geometry
    if source_geometry is None or source_geometry.is_empty:
        source_geometry = preview_result.path_geometry
    if source_geometry is None or source_geometry.is_empty:
        return []

    lines: list[base.BaseGeometry] = []
    if source_geometry.geom_type in {"LineString", "LinearRing"}:
        lines = [source_geometry]
    elif source_geometry.geom_type == "MultiLineString":
        lines = list(source_geometry.geoms)
    else:
        return []

    half_width = float(line_width) / 2.0
    ribbons: list[base.BaseGeometry] = []
    for line in lines:
        coords = np.asarray(line.coords, dtype=float)
        if coords.ndim != 2 or coords.shape[0] < 2:
            continue
        for start_point, end_point in zip(coords[:-1], coords[1:]):
            if np.linalg.norm(end_point - start_point) <= GeometrySettings.COORD_EPSILON:
                continue
            segment = LineString([tuple(start_point), tuple(end_point)])
            ribbon = segment.buffer(
                half_width,
                cap_style=2,
                join_style=2,
            )
            if ribbon.is_empty:
                continue
            ribbons.append(ribbon)
    return ribbons


def plot_pattern_result(preview_result, ax: plt.Axes | None = None) -> plt.Axes:
    """Render the main pattern preview, including walls and markers."""
    if ax is None:
        _, ax = plt.subplots(figsize=PlotSettings.RESULT_FIGSIZE)

    if preview_result.final_contour is not None:
        plot_geometry(
            ax,
            preview_result.final_contour,
            edge_only=True,
            linewidth=PlotSettings.CONTOUR_LINEWIDTH,
            color=PlotSettings.CONTOUR_COLOR,
        )
    if preview_result.infill_boundary is not None and not preview_result.infill_boundary.is_empty:
        plot_geometry(
            ax,
            preview_result.infill_boundary,
            edge_only=True,
            linewidth=PlotSettings.INFILL_BOUNDARY_LINEWIDTH,
            color=PlotSettings.INFILL_BOUNDARY_COLOR,
            linestyle=PlotSettings.INFILL_BOUNDARY_LINESTYLE,
        )

    for wall_geometry in preview_result.wall_geometries:
        plot_geometry(
            ax,
            wall_geometry,
            edge_only=True,
            linewidth=PlotSettings.WALL_LINEWIDTH,
            color=PlotSettings.WALL_COLOR,
            alpha=PlotSettings.WALL_ALPHA,
        )

    line_width = float(preview_result.line_width)
    linewidth_geometries: list[base.BaseGeometry] = []
    if line_width > 0:
        linewidth_geometries = _build_plot_linewidth_geometries(preview_result, line_width)

    if preview_result.path_geometry is not None and not preview_result.path_geometry.is_empty:
        plot_geometry(
            ax,
            preview_result.path_geometry,
            edge_only=True,
            linewidth=PlotSettings.CENTERLINE_LINEWIDTH,
            color=PlotSettings.CENTERLINE_COLOR,
            label=PlotSettings.LABEL_CENTERLINE,
            zorder=4,
        )
        first_linewidth_label = True
        for linewidth_geometry in linewidth_geometries:
            if linewidth_geometry is None or linewidth_geometry.is_empty:
                continue
            plot_geometry(
                ax,
                linewidth_geometry,
                edge_only=False,
                linewidth=0.0,
                color=PlotSettings.LINE_WIDTH_COLOR,
                alpha=PlotSettings.LINE_WIDTH_ALPHA,
                edgecolor="none",
                label=PlotSettings.LABEL_LINE_WIDTH if first_linewidth_label else None,
                zorder=2,
            )
            first_linewidth_label = False

    if preview_result.mesh_preview_geometry is not None and not preview_result.mesh_preview_geometry.is_empty:
        plot_geometry(
            ax,
            preview_result.mesh_preview_geometry,
            edge_only=True,
            linewidth=PlotSettings.MESH_PREVIEW_LINEWIDTH,
            color=PlotSettings.MESH_PREVIEW_COLOR,
            linestyle=PlotSettings.MESH_PREVIEW_LINESTYLE,
            label=PlotSettings.LABEL_MESH_PREVIEW,
        )

    markers = preview_result.markers
    if markers.get("path_start") is not None:
        ax.scatter(
            [markers["path_start"][0]],
            [markers["path_start"][1]],
            color=PlotSettings.START_COLOR,
            s=PlotSettings.START_MARKER_SIZE,
            zorder=5,
            label=PlotSettings.LABEL_START,
        )
    if markers.get("infill_start") is not None:
        ax.scatter(
            [markers["infill_start"][0]],
            [markers["infill_start"][1]],
            color=PlotSettings.INFILL_START_COLOR,
            s=PlotSettings.INFILL_START_MARKER_SIZE,
            zorder=6,
            edgecolors=PlotSettings.INFILL_START_EDGE_COLOR,
            linewidths=PlotSettings.INFILL_START_EDGE_LINEWIDTH,
            label=PlotSettings.LABEL_INFILL_START,
        )
    if markers.get("path_end") is not None:
        ax.scatter(
            [markers["path_end"][0]],
            [markers["path_end"][1]],
            color=PlotSettings.END_COLOR,
            s=PlotSettings.END_MARKER_SIZE,
            zorder=5,
            marker=PlotSettings.END_MARKER,
            label=PlotSettings.LABEL_END,
        )

    unit = str(preview_result.unit or ApiDefaults.UNIT)
    line_distance = float(preview_result.line_distance)
    fillet_radius_val = float(preview_result.fillet_radius)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"{preview_result.infill_pattern or 'pattern'} | "
        f"Walls={preview_result.num_walls}\n"
        f"Density~{preview_result.estimated_density:.{PlotSettings.TITLE_DENSITY_DECIMALS}f}% | "
        f"Distance={line_distance:.{PlotSettings.TITLE_DISTANCE_DECIMALS}f}{unit} | "
        f"Fillet>={fillet_radius_val:.{PlotSettings.TITLE_DISTANCE_DECIMALS}f}{unit}"
    )
    return ax


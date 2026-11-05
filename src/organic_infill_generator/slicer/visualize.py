"""Interactive and static visualization for slicer results."""
from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from .config import OPEN_LINE_COLOR
from .models import SlicerResult


def draw_layer(ax_raw: plt.Axes, ax_closed: plt.Axes, data: Dict, z: float) -> None:
    """Draw one layer: its raw segments and the contours closed from them."""
    ax_raw.clear()
    ax_closed.clear()

    ax_raw.set_title(
        f"RAW  Z={z:.2f}mm | Segs: {len(data['raw_segments'])}",
        fontsize=10, fontweight='bold')
    for seg in data['raw_segments']:
        ax_raw.plot([seg[0, 0], seg[1, 0]], [seg[0, 1], seg[1, 1]],
                    'b-', lw=1.0, alpha=0.7)

    ax_closed.set_title(
        f"CLOSED  Z={z:.2f}mm | Polys: {len(data['polygons'])} | {data['method']}",
        fontsize=10, fontweight='bold',
        color='#006600' if data['method'] != 'failed' else '#CC0000')
    for poly in data['polygons']:
        ax_closed.fill(*poly.exterior.xy, fc='gold', ec='black', alpha=0.6,
                       zorder=2, lw=1.0)
        for interior in poly.interiors:
            ax_closed.fill(*interior.xy, fc='white', ec='red', lw=1.0, zorder=3)
    for chain in data['open_lines']:
        ax_closed.plot(chain[:, 0], chain[:, 1], c=OPEN_LINE_COLOR, lw=2.5, zorder=6)

    for ax in (ax_raw, ax_closed):
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)


def plot_layer_viewer(layers_data: Dict[float, Dict], bounds_full: np.ndarray) -> None:
    """Dual-panel interactive viewer: RAW | CLOSED."""
    sorted_zs = sorted(layers_data.keys())
    if not sorted_zs:
        return

    fig, (ax_raw, ax_closed) = plt.subplots(1, 2, figsize=(16, 8))
    plt.subplots_adjust(bottom=0.25, wspace=0.3)

    minx, maxx = bounds_full[0, 0], bounds_full[1, 0]
    miny, maxy = bounds_full[0, 1], bounds_full[1, 1]
    margin = max(maxx - minx, maxy - miny) * 0.1

    def update(val):
        z = sorted_zs[int(val)]
        draw_layer(ax_raw, ax_closed, layers_data[z], z)
        for ax in (ax_raw, ax_closed):
            ax.set_xlim(minx - margin, maxx + margin)
            ax.set_ylim(miny - margin, maxy + margin)
        fig.canvas.draw_idle()

    ax_s = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_s, 'Layer', 0, len(sorted_zs) - 1, valinit=0, valstep=1)
    slider.on_changed(update)
    update(0)
    plt.show(block=True)


def show_contours(result: SlicerResult) -> None:
    """Open the interactive dual-panel viewer for a SlicerResult."""
    plot_layer_viewer(result._layers_data, result._mesh_bounds)

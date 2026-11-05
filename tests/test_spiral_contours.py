"""
Visual smoke test for the wall spiral connection in the new API.
"""
from __future__ import annotations


import matplotlib
from organic_infill_generator.infill_generator.visualization.plot_utils import finalize_figure, plot_pattern_result, should_force_agg

if should_force_agg():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


from organic_infill_generator.infill_generator.generator import _build_pattern_preview
from organic_infill_generator.infill_generator.settings import PlotSettings


def test_spiral_connections():
    contour_points = np.array([
        [0.0, 0.0], [5.0, 0.5], [6.0, 3.0],
        [4.0, 5.5], [1.0, 5.0], [-1.0, 2.5],
    ])

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    payloads = [
        {
            "title": "Spiral 3 walls",
            "infill_pattern": "honeycomb",
            "num_walls": 3,
            "line_width": 0.01,
            "infill_density": 20,
        },
        {
            "title": "Spiral 5 walls",
            "infill_pattern": "diamond",
            "num_walls": 5,
            "line_width": 0.01,
            "infill_density": 20,
        },
        {
            "title": "Spiral 7 walls",
            "infill_pattern": "sinusoidal",
            "num_walls": 7,
            "line_width": 0.01,
            "infill_density": 25,
        },
    ]

    for ax, payload in zip(axes, payloads):
        payload["contour_points"] = contour_points
        preview_result = _build_pattern_preview(payload)
        plot_pattern_result(preview_result, ax=ax)
        ax.legend(fontsize=PlotSettings.LEGEND_FONT_SIZE, loc=PlotSettings.LEGEND_LOCATION)

    plt.tight_layout()
    finalize_figure(fig, "spiral_test_debug.png")


if __name__ == "__main__":
    test_spiral_connections()

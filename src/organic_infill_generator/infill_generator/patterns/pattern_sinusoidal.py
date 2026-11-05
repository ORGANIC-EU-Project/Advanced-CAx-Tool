"""
pattern_sinusoidal.py
=====================
Sinusoidal column pattern generator.
"""
from __future__ import annotations

import numpy as np

from .pattern_base import DensityProfile, PatternBase, compute_step_breakpoints
from ..settings import GeometrySettings, SinusoidalSettings


class SinusoidalPattern(PatternBase):
    """Smooth sine-wave column pattern."""

    pattern_code = "SIN"

    def _generate_points(
            self,
            x_center: float,
            y_start: float,
            y_end: float,
            pattern_amplitude: float,
            pattern_step: float,
            density_profile: DensityProfile | None,
            line_width: float,
            column_anchor_y: float | None,
            column_anchor_phase: float | None,
    ) -> np.ndarray:
        if y_end - y_start <= 0:
            return np.array([[x_center, y_start], [x_center, y_end]], dtype=float)

        _EPS = GeometrySettings.AREA_EPSILON
        _SAMPLES = SinusoidalSettings.STEPS_PER_PATTERN_STEP
        _MIN_INC = SinusoidalSettings.MIN_INCREMENT

        # Macro breakpoints on the global grid.
        breakpoints = compute_step_breakpoints(
            y_start=y_start, y_end=y_end, pattern_step=pattern_step,
        )

        # Subdivide each macro step into sinusoidal samples.
        ys: list[float] = []
        for k in range(len(breakpoints) - 1):
            y_lo = breakpoints[k]
            y_hi = breakpoints[k + 1]
            h = y_hi - y_lo
            if h < _EPS:
                continue
            n_sub = max(1, int(round(h / pattern_step * _SAMPLES)))
            dy_sub = h / n_sub
            if dy_sub < _MIN_INC:
                n_sub = max(1, int(h / _MIN_INC))
                dy_sub = h / n_sub
            for j in range(n_sub):
                ys.append(y_lo + j * dy_sub)
        ys.append(breakpoints[-1])
        ys_arr = np.array(ys)

        # Half-wave per pattern_step: one full step = pi in phase.
        phases = np.full_like(ys_arr, -np.pi / 2.0, dtype=float)
        for i in range(1, len(ys_arr)):
            dy = ys_arr[i] - ys_arr[i - 1]
            local_step = (
                density_profile.evaluate_pattern_step(
                    x=x_center, y=ys_arr[i],
                    line_width=line_width,
                    default_pattern_step=pattern_step,
                )
                if density_profile
                else pattern_step
            )
            phases[i] = phases[i - 1] + np.pi * dy / max(local_step, SinusoidalSettings.MIN_PATTERN_STEP)

        phase_offset = 0.0
        if column_anchor_y is not None and ys_arr.size >= 2:
            anchor_y = float(np.clip(column_anchor_y, ys_arr[0], ys_arr[-1]))
            phase_at_anchor = float(np.interp(anchor_y, ys_arr, phases))
            phase_offset = float((column_anchor_phase or 0.0) - phase_at_anchor)

        half = pattern_amplitude / 2.0
        pts: list[list[float]] = []
        for phase_value, y_value in zip(phases, ys_arr):
            pts.append([x_center + half * np.sin(phase_value + phase_offset), y_value])

        return np.asarray(pts, dtype=float)


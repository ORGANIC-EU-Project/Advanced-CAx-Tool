from __future__ import annotations

from copy import deepcopy

from ..models import PatternRequest, PatternVariationRequest
from ..settings import DensitySettings


def _snap_to_step(value: float, step: float) -> float:
    snapped = round(value / step) * step
    return float(snapped)


def generate_density_variations(
        variation_request: PatternVariationRequest,
) -> list[PatternRequest]:
    """
    Generate balanced PatternRequest alternatives around the current global density.

    In the public API, the global density is stored as `infill_density`. This
    helper keeps that field as the source of truth, but also works against the
    `global_density` alias exposed by `PatternRequest`.

    Variations are:
    - centered around the original density
    - snapped to multiples of `step`
    - clamped to the public density bounds (0..100)
    - distributed as evenly as possible above and below the original value
    """
    original_pattern = variation_request.original_pattern
    n_variations = int(variation_request.n_variations)
    variation_range = float(variation_request.variation_range)
    step = float(variation_request.step)

    if n_variations < 0:
        raise ValueError("'n_variations' must be >= 0.")
    if n_variations == 0:
        return []
    if step <= 0:
        raise ValueError("'step' must be > 0.")
    if variation_range <= 0:
        raise ValueError("'variation_range' must be > 0.")

    min_density = float(DensitySettings.MIN_DENSITY_PERCENT)
    max_density = float(DensitySettings.MAX_DENSITY_PERCENT)
    base_density = float(original_pattern.global_density)

    max_offset_index = int(variation_range // step)
    if max_offset_index <= 0:
        raise ValueError("'variation_range' must allow at least one non-zero step.")

    below_candidates: list[float] = []
    above_candidates: list[float] = []
    seen_values = {float(base_density)}

    for offset_index in range(1, max_offset_index + 1):
        delta = offset_index * step

        below_value = _snap_to_step(base_density - delta, step)
        below_value = max(min_density, min(max_density, below_value))
        if below_value < base_density and below_value not in seen_values:
            below_candidates.append(float(below_value))
            seen_values.add(float(below_value))

        above_value = _snap_to_step(base_density + delta, step)
        above_value = max(min_density, min(max_density, above_value))
        if above_value > base_density and above_value not in seen_values:
            above_candidates.append(float(above_value))
            seen_values.add(float(above_value))

    if len(below_candidates) + len(above_candidates) < n_variations:
        raise ValueError(
            "Not enough unique density variations available within the configured range and bounds."
        )

    selected_densities: list[float] = []
    below_index = 0
    above_index = 0

    while len(selected_densities) < n_variations:
        if below_index < len(below_candidates):
            selected_densities.append(below_candidates[below_index])
            below_index += 1
            if len(selected_densities) >= n_variations:
                break

        if above_index < len(above_candidates):
            selected_densities.append(above_candidates[above_index])
            above_index += 1
            if len(selected_densities) >= n_variations:
                break

        if below_index >= len(below_candidates) and above_index >= len(above_candidates):
            break

    variations: list[PatternRequest] = []
    for density in selected_densities:
        variation = deepcopy(original_pattern)
        variation.infill_density = float(density)
        variations.append(variation)

    return variations

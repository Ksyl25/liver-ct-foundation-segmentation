"""Intensity normalization utilities for CT preprocessing."""

from __future__ import annotations

import numpy as np


def normalize_to_range(
    image: np.ndarray,
    output_min: float = 0.0,
    output_max: float = 1.0,
) -> np.ndarray:
    """Normalize an image to the requested output range."""

    if output_max <= output_min:
        raise ValueError("output_max must be greater than output_min.")

    array = np.asarray(image, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.full_like(array, output_min, dtype=np.float32)

    input_min = float(np.min(finite))
    input_max = float(np.max(finite))
    if np.isclose(input_min, input_max):
        return np.full_like(array, output_min, dtype=np.float32)

    normalized = (array - input_min) / (input_max - input_min)
    scaled = normalized * (output_max - output_min) + output_min
    scaled = np.nan_to_num(
        scaled,
        nan=output_min,
        posinf=output_max,
        neginf=output_min,
    )
    return np.clip(scaled, output_min, output_max).astype(np.float32)

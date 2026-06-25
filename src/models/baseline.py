"""Naive HU-threshold baseline segmentation.

This heuristic is intentionally simple and non-clinical. It thresholds HU
values that may include liver tissue, but it can also detect other soft
tissues and is not a robust segmentation model.
"""

from __future__ import annotations

import numpy as np


def _validate_thresholds(lower_hu: float, upper_hu: float) -> None:
    if lower_hu > upper_hu:
        raise ValueError("lower_hu must be less than or equal to upper_hu.")


def segment_liver_hu_threshold_slice(
    image_slice: np.ndarray,
    lower_hu: float = 40,
    upper_hu: float = 100,
) -> np.ndarray:
    """Segment a 2D slice with a naive liver HU threshold."""

    _validate_thresholds(lower_hu, upper_hu)
    image = np.asarray(image_slice, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image slice, got shape {image.shape}.")

    mask = (image >= lower_hu) & (image <= upper_hu)
    return mask.astype(np.uint8)


def segment_liver_hu_threshold(
    image_volume: np.ndarray,
    lower_hu: float = 40,
    upper_hu: float = 100,
) -> np.ndarray:
    """Segment a 3D volume with a naive liver HU threshold."""

    _validate_thresholds(lower_hu, upper_hu)
    image = np.asarray(image_volume, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image volume, got shape {image.shape}.")

    mask = (image >= lower_hu) & (image <= upper_hu)
    return mask.astype(np.uint8)

"""Hounsfield windowing utilities for CT display preprocessing."""

from __future__ import annotations

from typing import Final

import numpy as np

from src.preprocessing.normalization import normalize_to_range


WINDOW_PRESETS: Final[dict[str, dict[str, float]]] = {
    "liver": {"center": 60.0, "width": 150.0},
    "soft_tissue": {"center": 50.0, "width": 400.0},
    "bone": {"center": 300.0, "width": 1500.0},
    "lung": {"center": -600.0, "width": 1500.0},
}


def apply_window(image: np.ndarray, center: float, width: float) -> np.ndarray:
    """Clip a CT image to a Hounsfield window and normalize it to [0, 1]."""

    if width <= 0:
        raise ValueError("Window width must be greater than zero.")

    array = np.asarray(image, dtype=np.float32)
    lower = float(center) - float(width) / 2.0
    upper = float(center) + float(width) / 2.0
    clipped = np.clip(array, lower, upper)
    return normalize_to_range(clipped, output_min=0.0, output_max=1.0)


def apply_liver_window(image: np.ndarray) -> np.ndarray:
    """Apply the default liver CT window."""

    preset = WINDOW_PRESETS["liver"]
    return apply_window(
        image=image,
        center=preset["center"],
        width=preset["width"],
    )

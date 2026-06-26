"""Mask overlay utilities for 2D image slices."""

from __future__ import annotations

import numpy as np


MASK_TARGET_LABEL_MODE = "Target label only"
MASK_NONZERO_MODE = "All non-zero labels"


def normalize_slice_for_display(image_slice: np.ndarray) -> np.ndarray:
    """Normalize a 2D image slice to the [0, 1] display range."""

    image = np.asarray(image_slice, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image slice, got shape {image.shape}.")

    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)

    min_value = float(np.min(finite))
    max_value = float(np.max(finite))
    if np.isclose(max_value, min_value):
        return np.zeros_like(image, dtype=np.float32)

    normalized = (image - min_value) / (max_value - min_value)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def create_binary_mask_display(
    mask_slice: np.ndarray,
    target_label: int,
    mask_display_mode: str = MASK_TARGET_LABEL_MODE,
) -> np.ndarray:
    """Convert a raw mask slice to a binary display mask."""

    mask = np.asarray(mask_slice)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask slice, got shape {mask.shape}.")

    if mask_display_mode == MASK_TARGET_LABEL_MODE:
        binary_mask = mask == target_label
    elif mask_display_mode == MASK_NONZERO_MODE:
        binary_mask = mask > 0
    else:
        raise ValueError(f"Unsupported mask display mode: {mask_display_mode}")

    return binary_mask.astype(np.float32)


def create_mask_overlay(
    image_slice: np.ndarray,
    mask_slice: np.ndarray,
    alpha: float = 0.4,
    label: int | None = None,
) -> np.ndarray:
    """Create a red RGB overlay for a 2D mask on a grayscale image slice."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    image = np.asarray(image_slice)
    mask = np.asarray(mask_slice)
    if image.shape != mask.shape:
        raise ValueError(f"Image and mask slices must match, got {image.shape} and {mask.shape}.")
    if image.ndim != 2:
        raise ValueError(f"Expected 2D slices, got image shape {image.shape}.")

    normalized = normalize_slice_for_display(image)
    rgb = np.stack([normalized, normalized, normalized], axis=-1)

    mask_binary = mask > 0 if label is None else mask == label
    red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    rgb[mask_binary] = (1.0 - alpha) * rgb[mask_binary] + alpha * red

    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def draw_bbox_on_image(
    image_rgb: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
    color: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> np.ndarray:
    """Draw a bbox on a normalized RGB image without mutating input."""

    image = np.asarray(image_rgb, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an RGB image with shape (H, W, 3), got {image.shape}.")

    color_array = np.asarray(color, dtype=np.float32)
    if color_array.shape != (3,):
        raise ValueError("color must contain three RGB values.")
    color_array = np.clip(color_array, 0.0, 1.0)

    output = image.copy()
    if bbox is None:
        return np.clip(output, 0.0, 1.0).astype(np.float32)

    height, width, _ = output.shape
    x_min, y_min, x_max, y_max = bbox
    x_min = int(np.clip(x_min, 0, width - 1))
    x_max = int(np.clip(x_max, 0, width - 1))
    y_min = int(np.clip(y_min, 0, height - 1))
    y_max = int(np.clip(y_max, 0, height - 1))

    if x_min > x_max or y_min > y_max:
        return np.clip(output, 0.0, 1.0).astype(np.float32)

    output[y_min, x_min : x_max + 1] = color_array
    output[y_max, x_min : x_max + 1] = color_array
    output[y_min : y_max + 1, x_min] = color_array
    output[y_min : y_max + 1, x_max] = color_array

    return np.clip(output, 0.0, 1.0).astype(np.float32)

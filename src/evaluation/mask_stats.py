"""Mask statistics and slice lookup helpers."""

from __future__ import annotations

import numpy as np


def count_nonzero_pixels(mask: np.ndarray) -> int:
    """Return the number of positive pixels or voxels in a mask."""

    return int(np.count_nonzero(np.asarray(mask) > 0))


def mask_area(mask: np.ndarray) -> int:
    """Return the 2D/3D mask area as a count of positive pixels or voxels."""

    return count_nonzero_pixels(mask)


def find_slices_with_label(mask_volume: np.ndarray, label: int) -> list[int]:
    """Return axial slice indices that contain at least one voxel with label."""

    mask = np.asarray(mask_volume)
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask volume, got shape {mask.shape}.")

    slice_hits = np.any(mask == label, axis=(0, 1))
    return [int(index) for index in np.flatnonzero(slice_hits)]


def find_slices_with_nonzero_mask(mask_volume: np.ndarray) -> list[int]:
    """Return axial slice indices that contain at least one non-zero voxel."""

    mask = np.asarray(mask_volume)
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask volume, got shape {mask.shape}.")

    slice_hits = np.any(mask > 0, axis=(0, 1))
    return [int(index) for index in np.flatnonzero(slice_hits)]

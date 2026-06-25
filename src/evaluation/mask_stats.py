"""Mask statistics and slice lookup helpers."""

from __future__ import annotations

import numpy as np


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

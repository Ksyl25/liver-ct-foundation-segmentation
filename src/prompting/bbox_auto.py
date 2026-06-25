"""Automatic bounding box generation from binary or labeled masks."""

from __future__ import annotations

import numpy as np

BBox = tuple[int, int, int, int]


def get_bbox_from_mask(mask_slice: np.ndarray) -> BBox | None:
    """Return a 2D bbox as (x_min, y_min, x_max, y_max), or None if empty."""

    mask = np.asarray(mask_slice)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask slice, got shape {mask.shape}.")

    y_indices, x_indices = np.where(mask > 0)
    if x_indices.size == 0 or y_indices.size == 0:
        return None

    return (
        int(np.min(x_indices)),
        int(np.min(y_indices)),
        int(np.max(x_indices)),
        int(np.max(y_indices)),
    )


def get_bboxes_from_volume(mask_volume: np.ndarray) -> dict[int, BBox]:
    """Return bboxes for all non-empty axial slices in a 3D mask volume."""

    mask = np.asarray(mask_volume)
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask volume, got shape {mask.shape}.")

    bboxes: dict[int, BBox] = {}
    for slice_index in range(mask.shape[2]):
        bbox = get_bbox_from_mask(mask[:, :, slice_index])
        if bbox is not None:
            bboxes[int(slice_index)] = bbox

    return bboxes

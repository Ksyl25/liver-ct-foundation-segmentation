"""NIfTI loading helpers for Phase 1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


def load_nifti(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a NIfTI file and return the volume with basic metadata."""

    nifti_path = Path(path)
    if not nifti_path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {nifti_path}")
    if not nifti_path.is_file():
        raise ValueError(f"NIfTI path is not a file: {nifti_path}")

    image = nib.load(str(nifti_path))
    volume = np.asanyarray(image.dataobj)

    zooms = image.header.get_zooms()
    spacing = tuple(float(value) for value in zooms[: volume.ndim])

    finite_volume = volume[np.isfinite(volume)]
    if finite_volume.size:
        min_value = float(np.min(finite_volume))
        max_value = float(np.max(finite_volume))
        mean_value = float(np.mean(finite_volume))
    else:
        min_value = None
        max_value = None
        mean_value = None

    metadata: dict[str, Any] = {
        "path": str(nifti_path),
        "shape": tuple(int(value) for value in volume.shape),
        "dtype": str(volume.dtype),
        "affine": image.affine.tolist(),
        "spacing": spacing,
        "min": min_value,
        "max": max_value,
        "mean": mean_value,
    }
    return volume, metadata


def get_middle_slice_index(volume: np.ndarray) -> int:
    """Return the middle axial slice index for a 3D volume."""

    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}.")
    return int(volume.shape[2] // 2)


def validate_nifti_pair(image_volume: np.ndarray, mask_volume: np.ndarray) -> None:
    """Validate that image and mask volumes are compatible for Phase 1."""

    if image_volume.ndim != 3:
        raise ValueError(f"Image volume must be 3D, got shape {image_volume.shape}.")
    if mask_volume.ndim != 3:
        raise ValueError(f"Mask volume must be 3D, got shape {mask_volume.shape}.")
    if image_volume.shape != mask_volume.shape:
        raise ValueError(
            "Image and mask must have the same shape. "
            f"Got image {image_volume.shape} and mask {mask_volume.shape}."
        )

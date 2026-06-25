"""Simple local DICOM series loading for CT volumes.

This loader intentionally covers a small, local MVP use case. Production DICOM
sorting can be more complex because orientation, gantry tilt, multi-frame
objects and vendor-specific details may need additional handling.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np


class DicomLoadError(Exception):
    """Raised when a local DICOM series cannot be loaded safely."""


def _require_pydicom():
    try:
        import pydicom
    except ImportError:
        raise DicomLoadError(
            "pydicom is required to load DICOM series. Install requirements.txt."
        ) from None
    return pydicom


def _natural_sort_key(path: Path) -> list[Any]:
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def _as_safe_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float)):
        return value
    if hasattr(value, "__iter__"):
        try:
            return [float(item) if isinstance(item, (int, float)) else str(item) for item in value]
        except TypeError:
            return str(value)
    return str(value)


def _get_first_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        return value
    try:
        return value[0]
    except (TypeError, IndexError):
        return value


def _slice_position(dataset: Any) -> float | None:
    position = getattr(dataset, "ImagePositionPatient", None)
    if position is None or len(position) < 3:
        return None
    try:
        return float(position[2])
    except (TypeError, ValueError):
        return None


def _instance_number(dataset: Any) -> int | None:
    value = getattr(dataset, "InstanceNumber", None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def apply_rescale_to_hu(
    pixel_array: np.ndarray,
    slope: float | int | None,
    intercept: float | int | None,
) -> np.ndarray:
    """Apply DICOM rescale slope/intercept to produce approximate HU values."""

    slope_value = 1.0 if slope is None else float(slope)
    intercept_value = 0.0 if intercept is None else float(intercept)
    return (np.asarray(pixel_array, dtype=np.float32) * slope_value + intercept_value).astype(
        np.float32
    )


def sort_dicom_slices(datasets: list[Any]) -> list[Any]:
    """Sort DICOM slices by position, instance number, then natural file order."""

    if not datasets:
        return []

    positions = [_slice_position(dataset) for dataset in datasets]
    if all(position is not None for position in positions):
        return [
            dataset
            for _, dataset in sorted(
                zip(positions, datasets, strict=True),
                key=lambda item: item[0],
            )
        ]

    instance_numbers = [_instance_number(dataset) for dataset in datasets]
    if all(instance_number is not None for instance_number in instance_numbers):
        return [
            dataset
            for _, dataset in sorted(
                zip(instance_numbers, datasets, strict=True),
                key=lambda item: item[0],
            )
        ]

    return sorted(
        datasets,
        key=lambda dataset: _natural_sort_key(Path(getattr(dataset, "_source_path", ""))),
    )


def extract_dicom_metadata(datasets: list[Any], volume: np.ndarray) -> dict[str, Any]:
    """Extract safe DICOM metadata without exposing patient identifiers."""

    if not datasets:
        raise DicomLoadError("Cannot extract metadata from an empty DICOM series.")

    first = datasets[0]
    pixel_spacing = _as_safe_value(getattr(first, "PixelSpacing", None))
    slice_thickness = _as_safe_value(getattr(first, "SliceThickness", None))
    spacing = None
    if pixel_spacing is not None and slice_thickness is not None:
        try:
            spacing = [float(pixel_spacing[0]), float(pixel_spacing[1]), float(slice_thickness)]
        except (TypeError, ValueError, IndexError):
            spacing = None

    finite_volume = volume[np.isfinite(volume)]
    min_value = float(np.min(finite_volume)) if finite_volume.size else None
    max_value = float(np.max(finite_volume)) if finite_volume.size else None
    mean_value = float(np.mean(finite_volume)) if finite_volume.size else None

    sensitive_available = any(
        hasattr(first, field)
        for field in ("PatientName", "PatientID", "PatientBirthDate", "AccessionNumber")
    )

    return {
        "shape": tuple(int(value) for value in volume.shape),
        "dtype": str(volume.dtype),
        "min": min_value,
        "max": max_value,
        "mean": mean_value,
        "modality": _as_safe_value(getattr(first, "Modality", None)),
        "series_description": _as_safe_value(getattr(first, "SeriesDescription", None)),
        "study_description": _as_safe_value(getattr(first, "StudyDescription", None)),
        "pixel_spacing": pixel_spacing,
        "slice_thickness": slice_thickness,
        "spacing": spacing,
        "image_orientation_patient": _as_safe_value(
            getattr(first, "ImageOrientationPatient", None)
        ),
        "window_center": _as_safe_value(_get_first_value(getattr(first, "WindowCenter", None))),
        "window_width": _as_safe_value(_get_first_value(getattr(first, "WindowWidth", None))),
        "rescale_slope": _as_safe_value(getattr(first, "RescaleSlope", None)),
        "rescale_intercept": _as_safe_value(getattr(first, "RescaleIntercept", None)),
        "number_of_slices": int(len(datasets)),
        "patient_identifiers": "available / hidden" if sensitive_available else "not available",
        "privacy_note": "Clinical DICOM files may contain sensitive patient data.",
    }


def load_dicom_series(folder_path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a local folder of DICOM slices as a 3D numpy volume and metadata."""

    pydicom = _require_pydicom()
    folder = Path(folder_path)
    if not folder.exists():
        raise DicomLoadError(f"DICOM folder not found: {folder}")
    if not folder.is_dir():
        raise DicomLoadError(f"DICOM path is not a folder: {folder}")

    files = sorted((path for path in folder.iterdir() if path.is_file()), key=_natural_sort_key)
    if not files:
        raise DicomLoadError(f"DICOM folder is empty: {folder}")

    datasets = []
    for file_path in files:
        try:
            dataset = pydicom.dcmread(str(file_path), stop_before_pixels=False)
            _ = dataset.pixel_array
        except Exception:
            continue
        dataset._source_path = str(file_path)
        datasets.append(dataset)

    if not datasets:
        raise DicomLoadError(f"No valid DICOM slices found in folder: {folder}")

    sorted_datasets = sort_dicom_slices(datasets)
    slices = []
    for dataset in sorted_datasets:
        slope = getattr(dataset, "RescaleSlope", None)
        intercept = getattr(dataset, "RescaleIntercept", None)
        slices.append(apply_rescale_to_hu(dataset.pixel_array, slope, intercept))

    try:
        volume = np.stack(slices, axis=-1).astype(np.float32)
    except ValueError as exc:
        raise DicomLoadError("DICOM slices could not be stacked into a 3D volume.") from exc

    metadata = extract_dicom_metadata(sorted_datasets, volume)
    metadata["folder_path"] = str(folder)
    return volume, metadata

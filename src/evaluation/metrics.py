"""Evaluation metrics and CSV export helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.dice import dice_score

CSV_COLUMNS = [
    "case_id",
    "slice_index",
    "method",
    "target",
    "dice",
    "iou",
    "mask_area",
    "gt_area",
    "bbox",
    "bbox_source",
    "bbox_coverage",
    "inference_time",
    "status",
    "notes",
]


def _binary_mask(mask: np.ndarray, label: int | None = None) -> np.ndarray:
    array = np.asarray(mask)
    return array > 0 if label is None else array == label


def compute_dice(
    pred_mask: np.ndarray | None,
    gt_mask: np.ndarray | None,
    label: int | None = None,
) -> float | None:
    """Compute Dice, returning None if either mask is absent."""

    if pred_mask is None or gt_mask is None:
        return None
    return dice_score(pred_mask, gt_mask, label=label)


def compute_iou(
    pred_mask: np.ndarray | None,
    gt_mask: np.ndarray | None,
    label: int | None = None,
) -> float | None:
    """Compute intersection over union for binary or labeled masks."""

    if pred_mask is None or gt_mask is None:
        return None

    pred = _binary_mask(pred_mask, label)
    gt = _binary_mask(gt_mask, label)
    if pred.shape != gt.shape:
        raise ValueError(f"Mask shapes must match. Got {pred.shape} and {gt.shape}.")

    union = int(np.count_nonzero(pred | gt))
    if union == 0:
        return 1.0
    intersection = int(np.count_nonzero(pred & gt))
    return float(intersection / union)


def compute_slice_metrics(
    pred_slice: np.ndarray | None,
    gt_slice: np.ndarray | None,
    label: int = 1,
) -> dict[str, Any]:
    """Return simple metrics for one 2D prediction and ground-truth slice."""

    if pred_slice is None or gt_slice is None:
        return {
            "status": "not_available",
            "dice": None,
            "iou": None,
            "mask_area": None,
            "gt_area": None,
        }

    pred_binary = _binary_mask(pred_slice, label)
    gt_binary = _binary_mask(gt_slice, label)
    return {
        "status": "available",
        "dice": compute_dice(pred_slice, gt_slice, label=label),
        "iou": compute_iou(pred_slice, gt_slice, label=label),
        "mask_area": int(np.count_nonzero(pred_binary)),
        "gt_area": int(np.count_nonzero(gt_binary)),
    }


def compute_volume_metrics(
    pred_volume: np.ndarray | None,
    gt_volume: np.ndarray | None,
    label: int = 1,
) -> dict[str, Any]:
    """Return simple metrics for a 3D prediction and ground-truth volume."""

    if pred_volume is None or gt_volume is None:
        return {
            "status": "not_available",
            "dice": None,
            "iou": None,
            "mean_slice_dice_on_gt_slices": None,
        }

    pred = np.asarray(pred_volume)
    gt = np.asarray(gt_volume)
    if pred.shape != gt.shape:
        raise ValueError(f"Volume shapes must match. Got {pred.shape} and {gt.shape}.")
    if pred.ndim != 3:
        raise ValueError(f"Expected 3D volumes, got shape {pred.shape}.")

    gt_binary = _binary_mask(gt, label)
    slice_scores = []
    for slice_index in range(gt.shape[2]):
        if np.any(gt_binary[:, :, slice_index]):
            slice_scores.append(
                compute_dice(
                    pred[:, :, slice_index],
                    gt[:, :, slice_index],
                    label=label,
                )
            )

    return {
        "status": "available",
        "dice": compute_dice(pred, gt, label=label),
        "iou": compute_iou(pred, gt, label=label),
        "mean_slice_dice_on_gt_slices": (
            float(np.mean(slice_scores)) if slice_scores else None
        ),
    }


def compute_area_per_slice(
    mask_volume: np.ndarray,
    label: int | None = None,
) -> list[int]:
    """Return positive mask area for each axial slice."""

    mask = np.asarray(mask_volume)
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask volume, got shape {mask.shape}.")

    binary = _binary_mask(mask, label)
    return [
        int(np.count_nonzero(binary[:, :, slice_index]))
        for slice_index in range(binary.shape[2])
    ]


def compute_bbox_coverage(
    mask_slice: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
) -> float | None:
    """Return the fraction of positive mask pixels covered by an inclusive bbox."""

    if bbox is None:
        return None

    mask = _binary_mask(mask_slice, label=None)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask slice, got shape {mask.shape}.")

    total = int(np.count_nonzero(mask))
    if total == 0:
        return None

    height, width = mask.shape
    x_min, y_min, x_max, y_max = bbox
    x_min = int(np.clip(x_min, 0, width - 1))
    x_max = int(np.clip(x_max, 0, width - 1))
    y_min = int(np.clip(y_min, 0, height - 1))
    y_max = int(np.clip(y_max, 0, height - 1))
    if x_min > x_max or y_min > y_max:
        return 0.0

    covered = int(np.count_nonzero(mask[y_min : y_max + 1, x_min : x_max + 1]))
    return float(covered / total)


def export_metrics_to_csv(
    metrics: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Export metric rows to a CSV file and return the written path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    extra_columns = sorted(
        {
            key
            for row in metrics
            for key in row.keys()
            if key not in CSV_COLUMNS
        }
    )
    fieldnames = CSV_COLUMNS + extra_columns

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow({key: row.get(key) for key in fieldnames})

    return path

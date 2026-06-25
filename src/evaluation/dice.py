"""Dice score utilities for binary and labeled masks."""

from __future__ import annotations

import numpy as np


def dice_score(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    label: int | None = None,
    epsilon: float = 1e-8,
) -> float:
    """Compute the Dice score between two masks."""

    pred_array = np.asarray(pred_mask)
    gt_array = np.asarray(gt_mask)

    if pred_array.shape != gt_array.shape:
        raise ValueError(
            f"Mask shapes must match. Got {pred_array.shape} and {gt_array.shape}."
        )

    if label is None:
        pred_binary = pred_array > 0
        gt_binary = gt_array > 0
    else:
        pred_binary = pred_array == label
        gt_binary = gt_array == label

    pred_sum = int(np.count_nonzero(pred_binary))
    gt_sum = int(np.count_nonzero(gt_binary))

    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0

    intersection = int(np.count_nonzero(pred_binary & gt_binary))
    denominator = max(float(pred_sum + gt_sum), epsilon)
    score = (2.0 * intersection) / denominator
    return float(np.clip(score, 0.0, 1.0))


def slice_dice_score(
    pred_slice: np.ndarray,
    gt_slice: np.ndarray,
    label: int = 1,
) -> float:
    """Compute the Dice score for a single 2D slice."""

    return dice_score(pred_slice, gt_slice, label=label)

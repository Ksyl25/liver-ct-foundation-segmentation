import numpy as np

from src.evaluation.metrics import (
    compute_area_per_slice,
    compute_bbox_coverage,
    compute_dice,
    compute_iou,
    compute_slice_metrics,
    compute_volume_metrics,
)


def test_compute_dice_identical_masks_returns_one():
    mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)

    assert compute_dice(mask, mask) == 1.0


def test_compute_dice_no_overlap_returns_zero():
    pred = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    gt = np.array([[0, 0], [0, 1]], dtype=np.uint8)

    assert compute_dice(pred, gt) == 0.0


def test_compute_iou_identical_masks_returns_one():
    mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)

    assert compute_iou(mask, mask) == 1.0


def test_compute_iou_no_overlap_returns_zero():
    pred = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    gt = np.array([[0, 0], [0, 1]], dtype=np.uint8)

    assert compute_iou(pred, gt) == 0.0


def test_compute_slice_metrics_simple_case():
    pred = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    gt = np.array([[1, 0], [1, 0]], dtype=np.uint8)

    metrics = compute_slice_metrics(pred, gt, label=1)

    assert metrics["status"] == "available"
    assert metrics["dice"] == 0.5
    assert metrics["iou"] == 1 / 3
    assert metrics["mask_area"] == 2
    assert metrics["gt_area"] == 2


def test_compute_volume_metrics_simple_case():
    pred = np.zeros((2, 2, 2), dtype=np.uint8)
    gt = np.zeros((2, 2, 2), dtype=np.uint8)
    pred[0, 0, 0] = 1
    gt[0, 0, 0] = 1
    pred[1, 1, 1] = 1

    metrics = compute_volume_metrics(pred, gt, label=1)

    assert metrics["status"] == "available"
    assert metrics["dice"] == 2 / 3
    assert metrics["iou"] == 0.5
    assert metrics["mean_slice_dice_on_gt_slices"] == 1.0


def test_compute_area_per_slice_for_binary_and_label_masks():
    mask = np.zeros((2, 2, 3), dtype=np.uint8)
    mask[:, :, 0] = np.array([[0, 1], [2, 0]])
    mask[:, :, 1] = np.array([[1, 1], [0, 0]])

    assert compute_area_per_slice(mask) == [2, 2, 0]
    assert compute_area_per_slice(mask, label=1) == [1, 2, 0]


def test_compute_bbox_coverage():
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1:4, 1:4] = 1

    assert compute_bbox_coverage(mask, (1, 1, 3, 3)) == 1.0
    assert compute_bbox_coverage(mask, (1, 1, 2, 2)) == 4 / 9


def test_absent_prediction_metrics_are_not_available():
    gt = np.ones((2, 2), dtype=np.uint8)

    metrics = compute_slice_metrics(None, gt, label=1)

    assert metrics["status"] == "not_available"
    assert metrics["dice"] is None
    assert metrics["iou"] is None

import numpy as np

from src.evaluation.dice import dice_score, slice_dice_score


def test_dice_identical_binary_masks_returns_one():
    mask = np.array([[1, 0], [0, 1]])

    assert dice_score(mask, mask) == 1.0


def test_dice_no_overlap_returns_zero():
    pred = np.array([[1, 0], [0, 0]])
    gt = np.array([[0, 0], [0, 1]])

    assert dice_score(pred, gt) == 0.0


def test_dice_partial_overlap():
    pred = np.array([1, 1, 0, 0])
    gt = np.array([1, 0, 1, 0])

    assert dice_score(pred, gt) == 0.5


def test_dice_with_label_one():
    pred = np.array([[1, 2], [0, 1]])
    gt = np.array([[1, 0], [2, 1]])

    assert dice_score(pred, gt, label=1) == 1.0


def test_empty_masks_behavior():
    empty = np.zeros((2, 2), dtype=np.uint8)
    non_empty = np.array([[0, 1], [0, 0]], dtype=np.uint8)

    assert dice_score(empty, empty) == 1.0
    assert dice_score(empty, non_empty) == 0.0


def test_slice_dice_score_uses_label_one_by_default():
    pred = np.array([[1, 0], [0, 2]])
    gt = np.array([[1, 0], [2, 0]])

    assert slice_dice_score(pred, gt) == 1.0

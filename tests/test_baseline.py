import numpy as np
import pytest

from src.evaluation.dice import dice_score
from src.models.baseline import (
    segment_liver_hu_threshold,
    segment_liver_hu_threshold_slice,
)


def test_segment_liver_hu_threshold_slice_thresholds_inclusive_range():
    image = np.array([[39, 40, 60], [100, 101, -10]], dtype=np.float32)

    mask = segment_liver_hu_threshold_slice(image, lower_hu=40, upper_hu=100)

    expected = np.array([[0, 1, 1], [1, 0, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(mask, expected)


def test_segment_liver_hu_threshold_returns_binary_volume():
    image = np.array(
        [
            [[30, 50], [80, 120]],
            [[40, 100], [101, 39]],
        ],
        dtype=np.float32,
    )

    mask = segment_liver_hu_threshold(image, lower_hu=40, upper_hu=100)

    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()) == {0, 1}
    expected = np.array(
        [
            [[0, 1], [1, 0]],
            [[1, 1], [0, 0]],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(mask, expected)


def test_segment_liver_hu_threshold_rejects_invalid_threshold_order():
    image = np.zeros((2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="lower_hu"):
        segment_liver_hu_threshold_slice(image, lower_hu=100, upper_hu=40)


def test_baseline_dice_against_synthetic_ground_truth():
    image = np.array([[0, 50], [80, 120]], dtype=np.float32)
    gt = np.array([[0, 1], [1, 0]], dtype=np.uint8)

    pred = segment_liver_hu_threshold_slice(image, lower_hu=40, upper_hu=100)

    assert dice_score(pred, gt, label=1) == 1.0

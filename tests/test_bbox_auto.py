import numpy as np
import pytest

from src.prompting.bbox_auto import get_bbox_from_mask, get_bboxes_from_volume


def test_get_bbox_from_mask_returns_bbox_for_binary_slice():
    mask = np.zeros((5, 6), dtype=np.uint8)
    mask[1:4, 2:5] = 1

    assert get_bbox_from_mask(mask) == (2, 1, 4, 3)


def test_get_bbox_from_mask_returns_bbox_for_labeled_slice():
    mask = np.zeros((5, 6), dtype=np.uint8)
    mask[2, 1] = 2
    mask[4, 5] = 1

    assert get_bbox_from_mask(mask) == (1, 2, 5, 4)


def test_get_bbox_from_mask_returns_none_for_empty_slice():
    mask = np.zeros((5, 6), dtype=np.uint8)

    assert get_bbox_from_mask(mask) is None


def test_get_bboxes_from_volume_returns_non_empty_slices_only():
    volume = np.zeros((5, 6, 4), dtype=np.uint8)
    volume[1:4, 2:5, 1] = 1
    volume[0, 0, 3] = 2

    assert get_bboxes_from_volume(volume) == {
        1: (2, 1, 4, 3),
        3: (0, 0, 0, 0),
    }


def test_bbox_functions_reject_invalid_dimensions():
    with pytest.raises(ValueError, match="2D"):
        get_bbox_from_mask(np.zeros((2, 2, 2), dtype=np.uint8))

    with pytest.raises(ValueError, match="3D"):
        get_bboxes_from_volume(np.zeros((2, 2), dtype=np.uint8))

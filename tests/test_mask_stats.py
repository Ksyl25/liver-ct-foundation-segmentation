import numpy as np
import pytest

from src.evaluation.mask_stats import (
    find_slices_with_label,
    find_slices_with_nonzero_mask,
)


def test_find_slices_with_label_returns_matching_axial_indices():
    mask = np.zeros((3, 3, 5), dtype=np.uint8)
    mask[1, 1, 2] = 1
    mask[0, 0, 4] = 1
    mask[2, 2, 3] = 2

    assert find_slices_with_label(mask, label=1) == [2, 4]
    assert find_slices_with_label(mask, label=2) == [3]


def test_find_slices_with_nonzero_mask_returns_any_label_indices():
    mask = np.zeros((3, 3, 5), dtype=np.uint8)
    mask[1, 1, 1] = 1
    mask[0, 0, 3] = 2

    assert find_slices_with_nonzero_mask(mask) == [1, 3]


def test_find_slices_rejects_non_3d_masks():
    mask = np.zeros((3, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="3D"):
        find_slices_with_label(mask, label=1)
    with pytest.raises(ValueError, match="3D"):
        find_slices_with_nonzero_mask(mask)

import numpy as np
import pytest

from src.visualization.overlay import (
    MASK_NONZERO_MODE,
    MASK_TARGET_LABEL_MODE,
    create_binary_mask_display,
    draw_bbox_on_image,
)


def test_create_binary_mask_display_target_label_only():
    mask_slice = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)

    display = create_binary_mask_display(
        mask_slice,
        target_label=1,
        mask_display_mode=MASK_TARGET_LABEL_MODE,
    )

    expected = np.array([[0, 1, 0], [0, 1, 0]], dtype=np.float32)
    np.testing.assert_array_equal(display, expected)
    assert display.dtype == np.float32


def test_create_binary_mask_display_all_nonzero_labels():
    mask_slice = np.array([[0, 1, 2], [2, 0, 0]], dtype=np.uint8)

    display = create_binary_mask_display(
        mask_slice,
        target_label=1,
        mask_display_mode=MASK_NONZERO_MODE,
    )

    expected = np.array([[0, 1, 1], [1, 0, 0]], dtype=np.float32)
    np.testing.assert_array_equal(display, expected)


def test_create_binary_mask_display_rejects_invalid_mode():
    mask_slice = np.zeros((2, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="Unsupported"):
        create_binary_mask_display(mask_slice, target_label=1, mask_display_mode="raw")


def test_draw_bbox_on_image_draws_green_box_without_mutating_input():
    image = np.zeros((5, 5, 3), dtype=np.float32)
    bbox = (1, 1, 3, 3)

    output = draw_bbox_on_image(image, bbox)

    np.testing.assert_array_equal(image, np.zeros((5, 5, 3), dtype=np.float32))
    np.testing.assert_allclose(output[1, 1], np.array([0.0, 1.0, 0.0]))
    np.testing.assert_allclose(output[3, 3], np.array([0.0, 1.0, 0.0]))
    np.testing.assert_allclose(output[2, 2], np.array([0.0, 0.0, 0.0]))

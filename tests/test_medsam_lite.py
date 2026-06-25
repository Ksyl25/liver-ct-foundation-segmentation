import numpy as np
import pytest

from src.models.medsam_lite import (
    MedSAMLiteUnavailableError,
    detect_medsam_lite_api,
    get_runtime_status,
    load_medsam_lite_model,
    predict_slice_with_bbox,
    prepare_slice_for_medsam,
    validate_bbox,
)


def test_validate_bbox_accepts_valid_bbox():
    assert validate_bbox((1, 2, 4, 5), image_shape=(8, 10)) == (1, 2, 4, 5)


def test_validate_bbox_refuses_none():
    with pytest.raises(ValueError, match="None"):
        validate_bbox(None, image_shape=(8, 10))


def test_validate_bbox_refuses_out_of_bounds_bbox():
    with pytest.raises(ValueError, match="outside"):
        validate_bbox((1, 2, 10, 5), image_shape=(8, 10))


def test_validate_bbox_refuses_non_positive_extent():
    with pytest.raises(ValueError, match="x_max"):
        validate_bbox((1, 2, 1, 5), image_shape=(8, 10))
    with pytest.raises(ValueError, match="y_max"):
        validate_bbox((1, 2, 4, 2), image_shape=(8, 10))


def test_prepare_slice_for_medsam_converts_2d_slice_to_rgb():
    image = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)

    prepared = prepare_slice_for_medsam(image)

    assert prepared.shape == (2, 2, 3)
    assert prepared.dtype == np.float32
    assert float(np.min(prepared)) >= 0.0
    assert float(np.max(prepared)) <= 1.0
    np.testing.assert_allclose(prepared[:, :, 0], prepared[:, :, 1])
    np.testing.assert_allclose(prepared[:, :, 1], prepared[:, :, 2])


def test_prepare_slice_for_medsam_does_not_modify_input():
    image = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    original = image.copy()

    prepare_slice_for_medsam(image)

    np.testing.assert_array_equal(image, original)


def test_prepare_slice_for_medsam_refuses_invalid_image():
    image = np.zeros((2, 2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="2D"):
        prepare_slice_for_medsam(image)


def test_load_medsam_lite_model_raises_clear_error_if_checkpoint_absent(tmp_path):
    missing_checkpoint = tmp_path / "missing.pth"

    with pytest.raises(MedSAMLiteUnavailableError, match="checkpoint not found"):
        load_medsam_lite_model(missing_checkpoint)


def test_detect_medsam_lite_api_returns_status_dict():
    status = detect_medsam_lite_api()

    assert set(status) == {"available", "available_modules", "checked_modules"}
    assert isinstance(status["available"], bool)
    assert isinstance(status["available_modules"], list)
    assert "medsam_lite" in status["checked_modules"]


def test_get_runtime_status_reports_missing_checkpoint(tmp_path):
    status = get_runtime_status(tmp_path / "missing.pth")

    assert status["checkpoint_found"] is False
    assert "checkpoint_path" in status
    assert "torch_available" in status
    assert "medsam_api_available" in status
    assert "device_selected" in status


def test_predict_slice_with_bbox_refuses_bbox_none():
    image = np.zeros((8, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="None"):
        predict_slice_with_bbox(image, bbox=None)


def test_predict_slice_with_bbox_refuses_invalid_image():
    image = np.zeros((8, 8, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="2D"):
        predict_slice_with_bbox(image, bbox=(1, 1, 4, 4))


def test_predict_slice_with_bbox_requires_model_or_checkpoint():
    image = np.zeros((8, 8), dtype=np.float32)

    with pytest.raises(MedSAMLiteUnavailableError, match="checkpoint_path"):
        predict_slice_with_bbox(image, bbox=(1, 1, 4, 4))

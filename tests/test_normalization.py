import numpy as np
import pytest

from src.preprocessing.normalization import normalize_to_range


def test_normalize_to_default_range():
    image = np.array([10.0, 20.0, 30.0], dtype=np.float32)

    normalized = normalize_to_range(image)

    np.testing.assert_allclose(normalized, np.array([0.0, 0.5, 1.0]))


def test_normalize_to_custom_range():
    image = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    normalized = normalize_to_range(image, output_min=-1.0, output_max=1.0)

    np.testing.assert_allclose(normalized, np.array([-1.0, 0.0, 1.0]))


def test_normalize_constant_image_returns_output_min():
    image = np.full((2, 2), 42.0, dtype=np.float32)

    normalized = normalize_to_range(image, output_min=0.0, output_max=1.0)

    np.testing.assert_allclose(normalized, np.zeros((2, 2), dtype=np.float32))


def test_normalize_rejects_invalid_output_range():
    image = np.array([1.0, 2.0], dtype=np.float32)

    with pytest.raises(ValueError, match="output_max"):
        normalize_to_range(image, output_min=1.0, output_max=1.0)


def test_normalize_handles_non_finite_values():
    image = np.array([0.0, np.nan, np.inf, 10.0], dtype=np.float32)

    normalized = normalize_to_range(image)

    assert normalized.dtype == np.float32
    assert float(np.min(normalized)) >= 0.0
    assert float(np.max(normalized)) <= 1.0

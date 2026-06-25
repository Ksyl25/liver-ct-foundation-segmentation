import numpy as np
import pytest

from src.preprocessing.windowing import WINDOW_PRESETS, apply_liver_window, apply_window


def test_apply_window_clips_and_normalizes_values():
    image = np.array([-100.0, -15.0, 60.0, 135.0, 300.0], dtype=np.float32)

    windowed = apply_window(image, center=60, width=150)

    assert windowed.dtype == np.float32
    np.testing.assert_allclose(windowed, np.array([0.0, 0.0, 0.5, 1.0, 1.0]))


def test_apply_window_output_is_between_zero_and_one():
    image = np.linspace(-1000, 1000, 21, dtype=np.float32)

    windowed = apply_window(image, center=50, width=400)

    assert float(np.min(windowed)) >= 0.0
    assert float(np.max(windowed)) <= 1.0


def test_apply_window_rejects_invalid_width():
    image = np.array([0.0], dtype=np.float32)

    with pytest.raises(ValueError, match="width"):
        apply_window(image, center=60, width=0)


def test_apply_liver_window_uses_liver_preset():
    image = np.array([-15.0, 60.0, 135.0], dtype=np.float32)
    preset = WINDOW_PRESETS["liver"]

    liver_windowed = apply_liver_window(image)
    explicit_windowed = apply_window(
        image,
        center=preset["center"],
        width=preset["width"],
    )

    assert preset == {"center": 60.0, "width": 150.0}
    np.testing.assert_allclose(liver_windowed, explicit_windowed)


def test_required_window_presets_exist():
    assert WINDOW_PRESETS["liver"] == {"center": 60.0, "width": 150.0}
    assert WINDOW_PRESETS["soft_tissue"] == {"center": 50.0, "width": 400.0}
    assert WINDOW_PRESETS["bone"] == {"center": 300.0, "width": 1500.0}
    assert WINDOW_PRESETS["lung"] == {"center": -600.0, "width": 1500.0}

import nibabel as nib
import numpy as np
import pytest

from src.data.nifti_loader import (
    get_middle_slice_index,
    load_nifti,
    validate_nifti_pair,
)


def test_load_nifti_returns_volume_and_metadata(tmp_path):
    volume = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
    affine = np.diag([1.0, 1.5, 2.0, 1.0])
    nifti_path = tmp_path / "case.nii.gz"
    nib.save(nib.Nifti1Image(volume, affine), str(nifti_path))

    loaded_volume, metadata = load_nifti(nifti_path)

    assert loaded_volume.shape == volume.shape
    assert metadata["shape"] == volume.shape
    assert metadata["dtype"] == str(loaded_volume.dtype)
    assert metadata["spacing"] == (1.0, 1.5, 2.0)
    assert metadata["min"] == 0.0
    assert metadata["max"] == 23.0
    assert metadata["mean"] == pytest.approx(11.5)


def test_get_middle_slice_index():
    volume = np.zeros((8, 8, 5))

    assert get_middle_slice_index(volume) == 2


def test_validate_nifti_pair_rejects_shape_mismatch():
    image = np.zeros((2, 3, 4))
    mask = np.zeros((2, 3, 5))

    with pytest.raises(ValueError, match="same shape"):
        validate_nifti_pair(image, mask)


def test_validate_nifti_pair_rejects_non_3d_volumes():
    image = np.zeros((2, 3))
    mask = np.zeros((2, 3))

    with pytest.raises(ValueError, match="3D"):
        validate_nifti_pair(image, mask)

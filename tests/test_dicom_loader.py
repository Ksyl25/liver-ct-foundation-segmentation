from pathlib import Path

import numpy as np
import pytest

import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from src.data.dicom_loader import (
    DicomLoadError,
    apply_rescale_to_hu,
    load_dicom_series,
    sort_dicom_slices,
)


def _write_test_dicom(
    path: Path,
    pixel_array: np.ndarray,
    instance_number: int,
    slope: float = 1.0,
    intercept: float = 0.0,
    image_position_z: float | None = None,
) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "CT"
    dataset.SeriesDescription = "Synthetic CT"
    dataset.StudyDescription = "Synthetic Study"
    dataset.Rows, dataset.Columns = pixel_array.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    dataset.PixelSpacing = [0.7, 0.8]
    dataset.SliceThickness = 2.5
    dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    dataset.InstanceNumber = instance_number
    dataset.RescaleSlope = slope
    dataset.RescaleIntercept = intercept
    dataset.WindowCenter = 60
    dataset.WindowWidth = 150
    dataset.PatientName = "Hidden^Patient"
    dataset.PatientID = "123456"
    if image_position_z is not None:
        dataset.ImagePositionPatient = [0.0, 0.0, image_position_z]
    dataset.PixelData = pixel_array.astype(np.int16).tobytes()
    dataset.save_as(str(path), enforce_file_format=True)


def test_apply_rescale_to_hu_applies_slope_and_intercept():
    pixel_array = np.array([[0, 10]], dtype=np.int16)

    hu = apply_rescale_to_hu(pixel_array, slope=2, intercept=-1024)

    np.testing.assert_array_equal(hu, np.array([[-1024, -1004]], dtype=np.float32))


def test_load_dicom_series_reads_valid_folder_and_reconstructs_volume(tmp_path):
    _write_test_dicom(tmp_path / "slice_2.dcm", np.full((2, 2), 2), instance_number=2)
    _write_test_dicom(tmp_path / "slice_1.dcm", np.full((2, 2), 1), instance_number=1)
    (tmp_path / "notes.txt").write_text("not dicom", encoding="utf-8")

    volume, metadata = load_dicom_series(tmp_path)

    assert volume.shape == (2, 2, 2)
    np.testing.assert_array_equal(volume[:, :, 0], np.full((2, 2), 1, dtype=np.float32))
    np.testing.assert_array_equal(volume[:, :, 1], np.full((2, 2), 2, dtype=np.float32))
    assert metadata["modality"] == "CT"
    assert metadata["series_description"] == "Synthetic CT"
    assert metadata["number_of_slices"] == 2
    assert metadata["patient_identifiers"] == "available / hidden"
    assert "PatientName" not in metadata
    assert "PatientID" not in metadata


def test_load_dicom_series_applies_rescale(tmp_path):
    _write_test_dicom(
        tmp_path / "slice.dcm",
        np.array([[10, 20]], dtype=np.int16),
        instance_number=1,
        slope=2,
        intercept=-1000,
    )

    volume, metadata = load_dicom_series(tmp_path)

    np.testing.assert_array_equal(volume[:, :, 0], np.array([[-980, -960]], dtype=np.float32))
    assert metadata["rescale_slope"] == "2.0"
    assert metadata["rescale_intercept"] == "-1000.0"


def test_sort_dicom_slices_uses_image_position_when_available(tmp_path):
    _write_test_dicom(tmp_path / "a.dcm", np.zeros((2, 2)), instance_number=2, image_position_z=5)
    _write_test_dicom(tmp_path / "b.dcm", np.ones((2, 2)), instance_number=1, image_position_z=1)
    datasets = [pydicom.dcmread(str(tmp_path / "a.dcm")), pydicom.dcmread(str(tmp_path / "b.dcm"))]

    sorted_datasets = sort_dicom_slices(datasets)

    assert [int(dataset.InstanceNumber) for dataset in sorted_datasets] == [1, 2]


def test_load_dicom_series_raises_for_invalid_folder(tmp_path):
    missing_folder = tmp_path / "missing"

    with pytest.raises(DicomLoadError, match="not found"):
        load_dicom_series(missing_folder)


def test_load_dicom_series_raises_for_empty_folder(tmp_path):
    with pytest.raises(DicomLoadError, match="empty"):
        load_dicom_series(tmp_path)


def test_load_dicom_series_raises_when_no_valid_slices(tmp_path):
    (tmp_path / "notes.txt").write_text("not dicom", encoding="utf-8")

    with pytest.raises(DicomLoadError, match="No valid DICOM"):
        load_dicom_series(tmp_path)

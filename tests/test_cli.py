from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from src.cli import main


def _write_nifti(path: Path, volume: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(volume, np.eye(4)), str(path))


def _write_test_dicom(path: Path, pixel_array: np.ndarray, instance_number: int) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "CT"
    dataset.SeriesDescription = "CLI Synthetic CT"
    dataset.Rows, dataset.Columns = pixel_array.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    dataset.PixelSpacing = [1.0, 1.0]
    dataset.SliceThickness = 2.0
    dataset.InstanceNumber = instance_number
    dataset.RescaleSlope = 1.0
    dataset.RescaleIntercept = 0.0
    dataset.PixelData = pixel_array.astype(np.int16).tobytes()
    dataset.save_as(str(path), enforce_file_format=True)


def test_cli_run_viewer_prints_streamlit_command(capsys):
    exit_code = main(["run-viewer"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "streamlit run app/dashboard.py" in captured.out


def test_cli_inspect_nifti_with_synthetic_image_and_mask(tmp_path, capsys):
    image_path = tmp_path / "image.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    _write_nifti(image_path, np.arange(8, dtype=np.float32).reshape((2, 2, 2)))
    _write_nifti(mask_path, np.array([[[0, 1], [0, 0]], [[1, 0], [0, 0]]], dtype=np.uint8))

    exit_code = main(["inspect-nifti", "--image", str(image_path), "--mask", str(mask_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Inspect NIfTI" in captured.out
    assert "mask_unique_values" in captured.out
    assert "mask_nonzero_pixels: 2" in captured.out


def test_cli_evaluate_nifti_exports_csv(tmp_path, capsys):
    image_path = tmp_path / "image.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    output_csv = tmp_path / "metrics.csv"
    image = np.array([[[0, 50], [80, 120]], [[50, 0], [0, 0]]], dtype=np.float32)
    mask = ((image >= 40) & (image <= 100)).astype(np.uint8)
    _write_nifti(image_path, image)
    _write_nifti(mask_path, mask)

    exit_code = main(
        [
            "evaluate-nifti",
            "--image",
            str(image_path),
            "--mask",
            str(mask_path),
            "--output-csv",
            str(output_csv),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "hu_baseline_vs_gt_dice: 1.0" in captured.out
    assert output_csv.exists()


def test_cli_inspect_dicom_with_synthetic_series(tmp_path, capsys):
    _write_test_dicom(tmp_path / "slice_1.dcm", np.ones((2, 2)), instance_number=1)
    _write_test_dicom(tmp_path / "slice_2.dcm", np.ones((2, 2)) * 2, instance_number=2)

    exit_code = main(["inspect-dicom", "--dicom-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Inspect DICOM" in captured.out
    assert "series_description: CLI Synthetic CT" in captured.out
    assert "PatientName" not in captured.out
    assert "PatientID" not in captured.out


def test_cli_returns_error_for_missing_nifti(capsys):
    exit_code = main(["inspect-nifti", "--image", "missing.nii.gz"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Error:" in captured.err

# Usage

This project is designed for local Windows usage. It does not require cloud services and does not download medical data or model weights automatically.

## Windows setup

Python 3.11 is recommended, especially for the optional CUDA workflow. Python 3.10 is suitable for the core NIfTI, DICOM, Streamlit and test workflow.

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Optional PyTorch CUDA setup

MedSAM Lite inference requires a local PyTorch/CUDA environment, local LiteMedSAM dependencies and a local checkpoint. No checkpoint is included and no automatic weight download is performed.

Check CUDA availability:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

Place the MedSAM Lite checkpoint at:

```text
models/medsam_lite/medsam_lite.pth
```

The path can be changed in `config.yaml`:

```yaml
medsam:
  checkpoint_path: models/medsam_lite/medsam_lite.pth
  device: auto
  enabled: false
```

Do not commit model weights. The `models/` folder and common checkpoint extensions are ignored by Git.

MedSAM Lite inference is single-slice only in this project. Do not run full-volume inference on an RTX 3050 Laptop GPU unless a future implementation is explicitly optimized for that hardware.

## Launch Streamlit

```powershell
streamlit run app/dashboard.py
```

The app opens a local Streamlit dashboard with NIfTI and DICOM workflows.

## NIfTI image + mask workflow

Upload:

- a CT image volume in `.nii` or `.nii.gz` format;
- a matching mask in `.nii` or `.nii.gz` format.

The image and mask must be 3D volumes with the same shape. The default liver label is `1`.

In the slice viewer, use the display mode selector to switch between:

- Raw CT slice
- Liver windowed CT slice

The liver-windowed view uses a Hounsfield window with center `60` and width `150`. The mask and overlay remain available in both display modes.

The viewer also reports HU statistics:

- min
- max
- mean
- percentile 1
- percentile 99

## Mask diagnostics

The mask diagnostics panel shows:

- unique mask values in the full volume;
- unique mask values in the current slice;
- nonzero pixel counts;
- target-label pixel counts;
- label `1` and label `2` pixel counts;
- slice range where mask pixels exist.

Use this panel to check whether a black mask panel means an empty slice, a wrong label or an actual display issue.

## HU baseline and bbox

Enable:

- Show HU baseline segmentation
- Show automatic bounding box

The HU baseline is generated from the CT image only with a simple threshold. It is intentionally naive and is not clinical segmentation.

Bounding box sources:

- HU baseline bbox: automatic prompt
- Ground-truth mask bbox: debug only, not reportable as an automatic result

## MedSAM Lite

Open the `MedSAM Lite` section to inspect:

- checkpoint found or missing;
- torch available or unavailable;
- CUDA available or unavailable;
- selected GPU name when available;
- MedSAM API available or unavailable;
- selected device;
- selected bbox prompt source.

Click `Run MedSAM Lite on current slice` only when local dependencies and a checkpoint are available. If the local API or checkpoint is missing, the app shows a readable warning and does not display a fake prediction.

## MedSAM bbox comparison

The `MedSAM bbox comparison` table compares current-slice MedSAM Lite predictions using:

- HU baseline bbox: automatic prompt
- Ground-truth bbox: debug only, not reportable as an automatic result

The table reports:

- bbox source;
- bbox coordinates;
- Dice vs GT;
- IoU vs GT;
- inference time;
- notes.

MedSAM Lite performance is highly sensitive to bounding box quality. The comparison is single-slice only and must not be presented as dataset-level validation.

## Evaluation

Open the `Evaluation` section after uploading a ground-truth mask. The table reports current-slice metrics for:

- Ground-truth self-check
- HU baseline
- MedSAM Lite only if a real prediction exists

Columns include Dice, IoU, mask area, ground-truth area, bbox, bbox source, bbox coverage, status and notes.

Export current-slice metrics to:

```text
outputs/metrics/metrics_current_slice.csv
```

Generated outputs are ignored by Git.

## DICOM viewer

Open the `DICOM Viewer` section in Streamlit and enter a local folder containing one CT DICOM series. Click `Load DICOM series` to reconstruct a simple 3D volume and browse slices.

The DICOM viewer can display:

- raw / HU-converted slices;
- liver-windowed slices using the existing CT windowing;
- safe metadata such as modality, spacing, slice thickness and series description.

Patient identifiers such as patient name, patient ID, birth date and accession number are not displayed. DICOM files should still be anonymized before sharing.

The DICOM branch has no ground-truth mask in this MVP, so Dice evaluation is disabled for DICOM input. Use the NIfTI workflow for quantitative evaluation.

## CLI commands

Show the available commands:

```powershell
python scripts/cli.py --help
```

Print the viewer command:

```powershell
python scripts/cli.py run-viewer
```

Inspect a NIfTI image and optional mask:

```powershell
python scripts/cli.py inspect-nifti --image data/raw/nifti/imagesTr/liver_0.nii.gz --mask data/raw/nifti/labelsTr/liver_0.nii.gz
```

Inspect a local DICOM folder with safe metadata only:

```powershell
python scripts/cli.py inspect-dicom --dicom-dir data/raw/dicom/sample_series
```

Evaluate the naive HU baseline against a NIfTI ground-truth mask:

```powershell
python scripts/cli.py evaluate-nifti --image data/raw/nifti/imagesTr/liver_0.nii.gz --mask data/raw/nifti/labelsTr/liver_0.nii.gz --label 1
```

Export CLI evaluation metrics to CSV:

```powershell
python scripts/cli.py evaluate-nifti --image data/raw/nifti/imagesTr/liver_0.nii.gz --mask data/raw/nifti/labelsTr/liver_0.nii.gz --output-csv outputs/metrics/cli_metrics.csv
```

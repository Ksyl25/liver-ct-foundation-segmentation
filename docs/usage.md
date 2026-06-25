# Usage

Install dependencies from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Launch the viewer:

```powershell
streamlit run app/dashboard.py
```

Upload:

- a CT image volume in `.nii` or `.nii.gz` format
- a matching mask in `.nii` or `.nii.gz` format

The image and mask must be 3D volumes with the same shape.

In the slice viewer, use the display mode selector to switch between:

- Raw CT slice
- Liver windowed CT slice

The liver-windowed view uses a Hounsfield window with center `60` and width `150`. The mask and overlay remain available in both display modes. The metrics panel also shows basic HU statistics for the uploaded image volume.

To inspect the Phase 3 baseline, enable:

- Show HU baseline segmentation
- Show automatic bounding box

The baseline mask is generated from the CT image only, using a simple HU threshold. It does not use the uploaded ground-truth mask.

The bounding box source can be:

- HU baseline bbox: automatic bbox intended as a future MedSAM prompt source
- Ground-truth mask bbox (debug only): useful for checking mask alignment, not a real automatic prompt

When a ground-truth mask is uploaded, the app also shows baseline area, ground-truth area, current-slice Dice for baseline vs ground truth, bbox coordinates and bbox source.

## MedSAM Lite Local Setup

Phase 4A includes a safe MedSAM Lite readiness panel, but no model weights are distributed with this repository.

Place local weights here if you have them:

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

In Streamlit, open the `MedSAM Lite` section to check:

- checkpoint found or missing
- torch available or unavailable
- MedSAM API available or unavailable
- selected device
- selected bbox prompt source

Ground-truth bbox is available only for debugging. The baseline bbox is the intended automatic prompt source for future MedSAM Lite inference.

Click `Run MedSAM Lite on current slice` only after local dependencies and a checkpoint are available. If the local API or checkpoint is missing, the app will show a readable warning and will not display a fake prediction.

## Evaluation

Open the `Evaluation` section after uploading a ground-truth mask. The table reports current-slice metrics for:

- Ground-truth self-check
- HU baseline
- MedSAM Lite only if a real prediction exists

Columns include Dice, IoU, mask area, ground-truth area, bbox, bbox source, bbox coverage, status and notes. The HU baseline is a naive comparison point and is not clinical segmentation.

Use `Export current metrics to CSV` to write:

```text
outputs/metrics/metrics_current_slice.csv
```

MedSAM metrics remain `not_available` unless a real MedSAM Lite prediction has been produced.

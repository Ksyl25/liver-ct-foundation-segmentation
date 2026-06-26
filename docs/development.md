# Development

This document summarizes the local development workflow and repository hygiene rules.

## Project structure

```text
app/                  Streamlit application and UI components
src/data/             NIfTI and DICOM loaders
src/preprocessing/    CT windowing, normalization and slicing helpers
src/prompting/        Bounding box generation utilities
src/models/           HU baseline and MedSAM Lite wrapper
src/evaluation/       Dice, IoU, mask statistics and CSV export
src/visualization/    Mask overlays and bbox drawing
scripts/              CLI entrypoints
docs/                 Documentation
docs/screenshots/     README screenshots
tests/                Unit tests
data/                 Local data, ignored by Git
models/               Local checkpoints, ignored by Git
outputs/              Generated files, ignored by Git
```

## Run tests

From the repository root:

```powershell
python -m pytest
```

The tests cover the core local pipeline: NIfTI loading, DICOM loading, preprocessing, metrics, baseline, bbox helpers, overlays, CLI and MedSAM wrapper safety behavior.

## Run Streamlit

```powershell
streamlit run app/dashboard.py
```

Streamlit is the main interactive interface for NIfTI viewing, mask overlays, MedSAM single-slice checks, DICOM viewing and current-slice evaluation.

## Run CLI

```powershell
python scripts/cli.py --help
python scripts/cli.py run-viewer
python scripts/cli.py inspect-nifti --image data/raw/nifti/imagesTr/liver_0.nii.gz --mask data/raw/nifti/labelsTr/liver_0.nii.gz
python scripts/cli.py inspect-dicom --dicom-dir data/raw/dicom/sample_series
python scripts/cli.py evaluate-nifti --image data/raw/nifti/imagesTr/liver_0.nii.gz --mask data/raw/nifti/labelsTr/liver_0.nii.gz --label 1
```

## Repository hygiene

Do not commit:

- medical data;
- raw DICOM folders;
- NIfTI datasets;
- generated metrics;
- generated screenshots outside `docs/screenshots/`;
- model checkpoints;
- MedSAM Lite weights;
- local virtual environments;
- external model repositories.

The following paths are ignored by Git:

- `data/raw/`
- `data/processed/`
- `outputs/`
- `models/`
- `external/`
- `.venv/`
- `.venv311/`
- `.streamlit/`

The checkpoint path `models/medsam_lite/medsam_lite.pth` must remain local and untracked.

## Coding conventions

- Keep functions small and typed when they are part of the core pipeline.
- Prefer explicit errors over silent failures.
- Avoid hardcoded absolute Windows paths.
- Keep Streamlit UI logic thin and move reusable behavior into `src/`.
- Do not add heavy dependencies unless a phase explicitly requires them.
- Do not fake predictions, metrics or medical validation.
- Keep ground-truth bbox behavior clearly marked as debug only.

## Future improvement roadmap

The main next technical step is improving automatic prompt generation. Current results show that MedSAM Lite is functional, but segmentation quality is limited by the coarse HU baseline bounding box.

Useful next work:

- improve automatic bounding box generation;
- replace the naive HU-threshold baseline with a stronger liver localization method;
- add connected-component filtering;
- add anatomical constraints;
- test multi-slice prompt stability;
- evaluate on multiple MSD liver cases;
- compare HU baseline bbox, refined automatic bbox and debug bbox;
- investigate better preprocessing for MedSAM Lite input;
- add optional volume-level inference only if hardware allows;
- add DICOM-SEG export as a future interoperability extension;
- add expert review only in a proper medical setting.

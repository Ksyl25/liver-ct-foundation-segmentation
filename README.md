# Liver CT Foundation Segmentation

## Overview

A local medical imaging pipeline for liver CT segmentation using NIfTI preprocessing, Streamlit visualization and Dice-based evaluation. The current MVP focuses on loading a CT NIfTI volume and a matching NIfTI mask, browsing axial slices, displaying a mask overlay, and running a Dice self-check.

## Current MVP

- NIfTI image loading
- NIfTI mask loading
- Axial slice viewer
- Mask overlay
- Dice score self-check
- Raw and liver-windowed CT display
- Basic HU statistics
- Naive HU-threshold baseline segmentation
- Automatic 2D bounding boxes from baseline or mask
- MedSAM Lite integration scaffold with safe missing-checkpoint handling
- Current-slice evaluation table and CSV metrics export
- Simple local DICOM series ingestion and viewing
- Clean modular architecture

## Phase 2: CT windowing

Phase 2 adds minimal Hounsfield windowing for CT display and preparation. The Streamlit viewer can switch between a raw CT slice and a liver-windowed CT slice using center `60` and width `150`. The app also reports simple HU statistics for the uploaded image volume: min, max, mean, percentile 1 and percentile 99.

This is not automatic segmentation. It is a display and preprocessing step that prepares the project for later baseline and model inference phases.

## Phase 3: HU baseline segmentation and bounding boxes

Phase 3 adds a deliberately naive HU-threshold baseline using the default range `40` to `100` HU. It produces a binary mask that can be compared with the uploaded ground-truth mask and used to generate a simple automatic 2D bounding box on the selected slice.

This baseline is not clinical segmentation. It can detect other soft tissues and will not robustly isolate the liver. Ground-truth bounding boxes are available only as a debug/control view; the automatic future MedSAM prompt should come from the baseline or another prediction.

## Phase 4: MedSAM Lite integration scaffold

Phase 4A adds a safe MedSAM Lite wrapper and Streamlit readiness panel. It validates bounding boxes, prepares CT slices as pseudo-RGB inputs, checks local checkpoint status and handles missing weights or missing dependencies without crashing the app.

No MedSAM Lite weights are included, no automatic download is performed and no fine-tuning is implemented. Real inference is only available after Phase 4B wires a local MedSAM Lite API and a local checkpoint.

## Phase 4B: optional real MedSAM Lite inference

Phase 4B keeps MedSAM Lite optional. The app now reports whether local MedSAM/SAM-like APIs are importable, whether `torch` is available and whether the configured checkpoint exists. If any required piece is missing, Streamlit shows a controlled message and no prediction is displayed.

Real MedSAM Lite inference requires local dependencies, a local checkpoint configured in `config.yaml` and a supported local loader. No weights are included, no automatic download is performed and no Dice result is reported unless a real prediction exists.

## Phase 5: evaluation and metrics export

Phase 5 adds an educational evaluation layer for the current slice. Available metrics include Dice, IoU, mask area, ground-truth area, bbox coverage and CSV export to `outputs/metrics/`.

The HU baseline can be compared with the uploaded ground-truth mask. MedSAM Lite metrics are optional and are only reported if a real MedSAM prediction exists in the current session.

## Phase 6: DICOM ingestion

Phase 6 adds simple local DICOM CT series loading. It reads a folder of DICOM slices, reconstructs a 3D volume, applies `RescaleSlope` and `RescaleIntercept` when available, extracts safe metadata and displays slices in Streamlit.

DICOM support is for local ingestion and visualization only. Quantitative evaluation remains based on NIfTI image and mask pairs with ground truth.

## Roadmap

- Phase 2: Hounsfield windowing and CT preprocessing
- Phase 3: baseline segmentation and automatic bounding boxes
- Phase 4A: MedSAM Lite integration scaffold
- Phase 4B: MedSAM Lite zero-shot inference with local weights
- Phase 5: evaluation and metrics export
- Phase 6: DICOM ingestion
- Phase 7: CLI
- Phase 8: premium documentation and screenshots

## Installation

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

```powershell
streamlit run app/dashboard.py
```

Upload a CT image file in `.nii` or `.nii.gz` format and a compatible mask file in `.nii` or `.nii.gz` format. The image and mask must have the same shape.

## Data

Medical data is not included in this repository. Provide your own local NIfTI CT volume and matching NIfTI mask. Do not commit medical data, generated masks, or patient-related files.

## Disclaimer

Educational project only. Not for clinical use or diagnosis.

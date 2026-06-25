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
- Clean modular architecture

## Phase 2: CT windowing

Phase 2 adds minimal Hounsfield windowing for CT display and preparation. The Streamlit viewer can switch between a raw CT slice and a liver-windowed CT slice using center `60` and width `150`. The app also reports simple HU statistics for the uploaded image volume: min, max, mean, percentile 1 and percentile 99.

This is not automatic segmentation. It is a display and preprocessing step that prepares the project for later baseline and model inference phases.

## Roadmap

- Phase 2: Hounsfield windowing and CT preprocessing
- Phase 3: baseline segmentation and automatic bounding boxes
- Phase 4: MedSAM Lite zero-shot inference
- Phase 5: benchmark and evaluation
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

# Limitations

This repository is an educational / portfolio project. It is not clinically validated and is not for diagnosis.

## Clinical scope

- Not clinically validated.
- Not for diagnosis.
- Educational / portfolio project only.
- No expert review has been performed.
- No claim is made about clinical safety, reliability or deployment readiness.
- Current metrics are not medical reports.

## Segmentation scope

- The HU baseline is naive.
- HU thresholding alone does not truly segment the liver.
- The HU baseline can detect other organs, vessels and soft tissues in the same intensity range.
- The HU baseline bbox is often too coarse.
- MedSAM Lite performance is highly sensitive to bounding box quality.
- Ground-truth bbox is debug only and must not be reported as an automatic result.
- Current MedSAM Lite results are preliminary single-slice observations.
- No dataset-level validation yet.
- No fine-tuning.

## MedSAM scope

- MedSAM/LiteMedSAM model inference is optional and local-only.
- MedSAM Lite weights are not distributed with this repository.
- No MedSAM checkpoint is downloaded automatically.
- MedSAM Lite inference depends on local installation, local weights and hardware.
- MedSAM Lite inference also depends on a supported local API being available to Python.
- Single-slice MedSAM inference only.
- No full-volume MedSAM inference.
- No batch inference.
- No 3D MedSAM reconstruction.
- Laptop GPUs such as RTX 3050 Laptop GPU may run out of memory; CUDA OOM is handled as a user-facing error where possible.

## Data and preprocessing

- Medical data is not included.
- Assumes image and mask share the same shape for NIfTI evaluation.
- Does not perform advanced orientation normalization.
- Does not resample volumes to a common spacing.
- CT windowing is available for display and preparation, but it is not automatic segmentation.
- Better preprocessing for MedSAM Lite input remains future work.

## DICOM scope

- DICOM support is intentionally simple and local.
- DICOM support may not cover all vendor-specific edge cases.
- No PACS integration.
- No DICOM-SEG export.
- The DICOM branch has no Dice evaluation unless a matching mask is added in a future phase.
- DICOM files should be anonymized before sharing, even though the app avoids displaying common patient identifiers.

## Deployment scope

- No cloud deployment.
- No FastAPI service.
- No production monitoring.
- No security hardening for clinical environments.

## CLI scope

- The CLI is a lightweight local helper and does not replace clinical validation.
- CLI evaluation still uses the naive HU baseline and optional local MedSAM status only.

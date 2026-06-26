# Screenshots for README

Put these files in `docs/screenshots/` in the repository.

Recommended README order:

1. `01_nifti_viewer.png` — NIfTI CT slice, mask, overlay.
2. `02_hu_baseline_bbox.png` — HU baseline segmentation and automatic bbox.
3. `03_medsam_bbox_comparison.png` — HU baseline bbox vs ground-truth bbox debug comparison.
4. `04_medsam_comparison_results.png` — MedSAM bbox comparison results table.
5. `05_evaluation_metrics.png` — Evaluation metrics and CSV export panel.
6. `06_dicom_viewer.png` — Real DICOM CT viewer.

Notes:
- Ground-truth bbox must stay labeled as debug only.
- Results are preliminary single-slice results, not dataset-level validation.
- DICOM screenshot demonstrates local ingestion only; Dice is disabled for DICOM because no matching GT mask is provided.

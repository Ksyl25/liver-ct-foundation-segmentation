# Limitations

- No clinical validation.
- No model inference yet.
- No DICOM support yet.
- CT windowing is available for display and preparation, but it is not automatic segmentation.
- HU thresholding alone does not truly segment the liver.
- The HU baseline can detect other organs, vessels and soft tissues in the same intensity range.
- Assumes image and mask share the same shape.
- Does not reorient or resample input volumes.
- Automatic bounding boxes are simple 2D boxes from masks or the HU baseline.
- MedSAM Lite weights are not distributed with this repository.
- MedSAM Lite inference depends on local installation, local weights and hardware.
- MedSAM Lite inference also depends on a supported local API being available to Python.
- CPU MedSAM Lite inference may be slow once real inference is implemented.
- No MedSAM Lite fine-tuning is implemented.
- MedSAM quality depends strongly on the selected bounding box.
- Current metrics on a single case or slice are not representative of clinical validation.
- CSV exports are local educational artifacts and should not be treated as medical reports.

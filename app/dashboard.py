"""Streamlit dashboard for the Phase 1 liver CT NIfTI MVP."""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.metrics_panel import (
    render_dice_metrics,
    render_hu_statistics,
    render_metadata,
)
from app.components.dicom_viewer import render_dicom_viewer
from app.components.sidebar import render_sidebar
from app.components.viewer import render_slice_viewer
from src.data.nifti_loader import load_nifti, validate_nifti_pair


def _uploaded_nifti_suffix(file_name: str) -> str:
    lower_name = file_name.lower()
    if lower_name.endswith(".nii.gz"):
        return ".nii.gz"
    if lower_name.endswith(".nii"):
        return ".nii"
    return ".nii"


def _load_uploaded_nifti(uploaded_file):
    suffix = _uploaded_nifti_suffix(uploaded_file.name)
    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_path = Path(temp_file.name)

        return load_nifti(temp_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def main() -> None:
    """Run the Streamlit dashboard."""

    st.set_page_config(
        page_title="Liver CT Foundation Segmentation",
        layout="wide",
    )

    st.title("Liver CT Foundation Segmentation")
    st.warning("Educational project only. Not for clinical use or diagnosis.")

    render_dicom_viewer()

    uploaded_image, uploaded_mask, target_label = render_sidebar()

    if uploaded_image is None:
        st.info("Upload a CT NIfTI image to start.")
        return

    image_volume = None
    mask_volume = None

    try:
        image_volume, image_metadata = _load_uploaded_nifti(uploaded_image)
    except Exception as exc:
        st.error(f"Could not load image NIfTI: {exc}")
        return

    mask_metadata = None
    if uploaded_mask is not None:
        try:
            mask_volume, mask_metadata = _load_uploaded_nifti(uploaded_mask)
            validate_nifti_pair(image_volume, mask_volume)
        except Exception as exc:
            st.error(f"Could not load or validate mask NIfTI: {exc}")
            return
    else:
        st.info("Upload a matching mask NIfTI to enable overlay and Dice metrics.")

    metadata_col, metrics_col = st.columns([2, 1])
    with metadata_col:
        render_metadata(image_metadata, "Image metadata")
        if mask_metadata is not None:
            render_metadata(mask_metadata, "Mask metadata")
    with metrics_col:
        render_hu_statistics(image_volume)
        render_dice_metrics(mask_volume, target_label)

    st.divider()
    render_slice_viewer(image_volume, mask_volume, target_label)


if __name__ == "__main__":
    main()

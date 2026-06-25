"""Streamlit component for simple local DICOM series viewing."""

from __future__ import annotations

import numpy as np
import streamlit as st

from src.data.dicom_loader import DicomLoadError, load_dicom_series
from src.preprocessing.windowing import apply_liver_window
from src.utils.config import load_config
from src.visualization.overlay import normalize_slice_for_display

DICOM_RAW_MODE = "Raw / HU DICOM slice"
DICOM_LIVER_WINDOW_MODE = "Liver windowed DICOM slice"


def _default_dicom_folder() -> str:
    config = load_config()
    return str(config.get("data", {}).get("dicom_folder", "data/raw/dicom/sample_series"))


def _render_safe_metadata(metadata: dict) -> None:
    safe_keys = [
        "folder_path",
        "shape",
        "dtype",
        "min",
        "max",
        "mean",
        "modality",
        "series_description",
        "study_description",
        "pixel_spacing",
        "slice_thickness",
        "spacing",
        "image_orientation_patient",
        "window_center",
        "window_width",
        "rescale_slope",
        "rescale_intercept",
        "number_of_slices",
        "patient_identifiers",
        "privacy_note",
    ]
    st.json({key: metadata.get(key) for key in safe_keys})


def render_dicom_viewer() -> None:
    """Render a simple local DICOM series viewer."""

    with st.expander("DICOM Viewer", expanded=False):
        st.warning(
            "Clinical DICOM files may contain sensitive patient data. Patient "
            "identifiers are not displayed here. Use anonymized data before sharing."
        )
        st.info(
            "No ground-truth mask is available for DICOM input, so Dice evaluation "
            "is disabled."
        )

        folder_path = st.text_input(
            "Local DICOM folder",
            value=_default_dicom_folder(),
            help="Enter a local folder containing one DICOM CT series.",
        )
        display_mode = st.radio(
            "DICOM display mode",
            options=[DICOM_RAW_MODE, DICOM_LIVER_WINDOW_MODE],
            horizontal=True,
        )

        if not st.button("Load DICOM series"):
            return

        try:
            volume, metadata = load_dicom_series(folder_path)
        except DicomLoadError as exc:
            st.warning(str(exc))
            return

        st.subheader("Safe DICOM metadata")
        _render_safe_metadata(metadata)

        if volume.ndim != 3:
            st.error(f"Expected a 3D DICOM volume, got shape {volume.shape}.")
            return

        max_index = volume.shape[2] - 1
        slice_index = st.slider(
            "DICOM axial slice",
            min_value=0,
            max_value=max_index,
            value=max_index // 2,
            step=1,
        )
        st.caption(f"DICOM slice index: {slice_index} / {max_index}")

        image_slice = volume[:, :, slice_index]
        if display_mode == DICOM_LIVER_WINDOW_MODE:
            display_slice = apply_liver_window(image_slice)
        else:
            display_slice = image_slice

        st.image(
            normalize_slice_for_display(np.asarray(display_slice)),
            caption=display_mode,
            clamp=True,
        )

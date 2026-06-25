"""Sidebar controls for the Streamlit NIfTI viewer."""

from __future__ import annotations

import streamlit as st


def render_sidebar():
    """Render upload controls and return image, mask and target label."""

    with st.sidebar:
        st.title("Liver CT MVP")
        st.write(
            "Upload a CT NIfTI volume and a matching liver mask to inspect axial slices."
        )
        uploaded_image = st.file_uploader(
            "CT image NIfTI",
            type=["nii", "nii.gz"],
            accept_multiple_files=False,
        )
        uploaded_mask = st.file_uploader(
            "Mask NIfTI",
            type=["nii", "nii.gz"],
            accept_multiple_files=False,
        )
        target_label = st.number_input(
            "Target label",
            min_value=0,
            value=1,
            step=1,
            help="Label value used for the liver mask.",
        )

    return uploaded_image, uploaded_mask, int(target_label)

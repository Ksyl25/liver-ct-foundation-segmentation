"""Slice viewer component for CT volumes and masks."""

from __future__ import annotations

import numpy as np
import streamlit as st

from src.data.nifti_loader import get_middle_slice_index
from src.visualization.overlay import create_mask_overlay, normalize_slice_for_display


def render_slice_viewer(
    image_volume: np.ndarray,
    mask_volume: np.ndarray | None,
    target_label: int,
) -> None:
    """Render an axial slice viewer for a CT image and optional mask."""

    if image_volume.ndim != 3:
        st.error("The image volume must be 3D.")
        return

    default_index = get_middle_slice_index(image_volume)
    max_index = image_volume.shape[2] - 1
    slice_index = st.slider(
        "Axial slice",
        min_value=0,
        max_value=max_index,
        value=default_index,
        step=1,
    )

    st.caption(f"Slice index: {slice_index} / {max_index}")

    image_slice = image_volume[:, :, slice_index]
    normalized_image = normalize_slice_for_display(image_slice)

    if mask_volume is None:
        st.image(normalized_image, caption="CT image", clamp=True)
        st.info("Upload a matching mask to display mask and overlay views.")
        return

    mask_slice = mask_volume[:, :, slice_index]
    mask_display = (mask_slice == target_label).astype(float)
    overlay = create_mask_overlay(
        image_slice=image_slice,
        mask_slice=mask_slice,
        alpha=0.4,
        label=target_label,
    )

    image_col, mask_col, overlay_col = st.columns(3)
    with image_col:
        st.image(normalized_image, caption="CT image", clamp=True)
    with mask_col:
        st.image(mask_display, caption="Mask", clamp=True)
    with overlay_col:
        st.image(overlay, caption="Overlay", clamp=True)

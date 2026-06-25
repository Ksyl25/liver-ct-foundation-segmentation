"""Slice viewer component for CT volumes and masks."""

from __future__ import annotations

import numpy as np
import streamlit as st

from src.data.nifti_loader import get_middle_slice_index
from src.evaluation.mask_stats import (
    find_slices_with_label,
    find_slices_with_nonzero_mask,
)
from src.preprocessing.windowing import apply_liver_window
from src.visualization.overlay import (
    MASK_NONZERO_MODE,
    MASK_TARGET_LABEL_MODE,
    create_binary_mask_display,
    create_mask_overlay,
    normalize_slice_for_display,
)


RAW_DISPLAY_MODE = "Raw CT slice"
LIVER_WINDOW_DISPLAY_MODE = "Liver windowed CT slice"
SLICE_STATE_KEY = "axial_slice_index"


def _format_unique_values(values: np.ndarray, max_values: int = 20) -> str:
    unique_values = np.unique(values)
    shown_values = unique_values[:max_values].tolist()
    suffix = "" if unique_values.size <= max_values else f" ... ({unique_values.size} total)"
    return f"{shown_values}{suffix}"


def _format_slice_range(indices: list[int]) -> str:
    if not indices:
        return "No mask pixels found."
    return f"{indices[0]} to {indices[-1]}"


def _render_mask_navigation(
    mask_volume: np.ndarray,
    target_label: int,
    mask_display_mode: str,
) -> None:
    nonzero_slices = find_slices_with_nonzero_mask(mask_volume)
    target_slices = find_slices_with_label(mask_volume, target_label)

    st.caption(f"First/last slice with any mask: {_format_slice_range(nonzero_slices)}")
    st.caption(
        f"First/last slice with target label {target_label}: "
        f"{_format_slice_range(target_slices)}"
    )

    active_slices = (
        target_slices if mask_display_mode == MASK_TARGET_LABEL_MODE else nonzero_slices
    )
    if active_slices and st.button("Jump to first mask slice"):
        st.session_state[SLICE_STATE_KEY] = active_slices[0]


def _render_mask_diagnostics(
    mask_volume: np.ndarray,
    mask_slice: np.ndarray,
    binary_mask: np.ndarray,
    target_label: int,
) -> None:
    with st.expander("Mask diagnostics", expanded=True):
        st.write(
            {
                "unique_values_volume": _format_unique_values(mask_volume),
                "unique_values_current_slice": _format_unique_values(mask_slice),
                "nonzero_pixels_volume": int(np.count_nonzero(mask_volume)),
                "nonzero_pixels_current_slice": int(np.count_nonzero(mask_slice)),
                f"target_label_{target_label}_pixels_current_slice": int(
                    np.count_nonzero(mask_slice == target_label)
                ),
                "label_1_pixels_current_slice": int(np.count_nonzero(mask_slice == 1)),
                "label_2_pixels_current_slice": int(np.count_nonzero(mask_slice == 2)),
                "displayed_mask_pixels_current_slice": int(np.count_nonzero(binary_mask)),
            }
        )


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
    display_mode = st.radio(
        "Display mode",
        options=[RAW_DISPLAY_MODE, LIVER_WINDOW_DISPLAY_MODE],
        horizontal=True,
    )
    mask_display_mode = MASK_TARGET_LABEL_MODE
    if mask_volume is not None:
        mask_display_mode = st.radio(
            "Mask display mode",
            options=[MASK_TARGET_LABEL_MODE, MASK_NONZERO_MODE],
            horizontal=True,
        )
        _render_mask_navigation(mask_volume, target_label, mask_display_mode)

    if SLICE_STATE_KEY not in st.session_state:
        st.session_state[SLICE_STATE_KEY] = default_index
    st.session_state[SLICE_STATE_KEY] = int(
        np.clip(st.session_state[SLICE_STATE_KEY], 0, max_index)
    )

    slice_index = st.slider(
        "Axial slice",
        min_value=0,
        max_value=max_index,
        step=1,
        key=SLICE_STATE_KEY,
    )

    st.caption(f"Slice index: {slice_index} / {max_index}")

    image_slice = image_volume[:, :, slice_index]
    if display_mode == LIVER_WINDOW_DISPLAY_MODE:
        display_slice = apply_liver_window(image_slice)
    else:
        display_slice = image_slice

    normalized_image = normalize_slice_for_display(display_slice)

    if mask_volume is None:
        st.image(normalized_image, caption=display_mode, clamp=True)
        st.info("Upload a matching mask to display mask and overlay views.")
        return

    mask_slice = mask_volume[:, :, slice_index]
    mask_display = create_binary_mask_display(
        mask_slice,
        target_label=target_label,
        mask_display_mode=mask_display_mode,
    )
    overlay = create_mask_overlay(
        image_slice=display_slice,
        mask_slice=mask_display,
        alpha=0.4,
        label=None,
    )

    _render_mask_diagnostics(mask_volume, mask_slice, mask_display, target_label)

    image_col, mask_col, overlay_col = st.columns(3)
    with image_col:
        st.image(normalized_image, caption=display_mode, clamp=True)
    with mask_col:
        st.image(mask_display, caption="Mask", clamp=True)
    with overlay_col:
        st.image(overlay, caption="Overlay", clamp=True)

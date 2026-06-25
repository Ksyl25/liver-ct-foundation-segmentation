"""Slice viewer component for CT volumes and masks."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import streamlit as st

from src.data.nifti_loader import get_middle_slice_index
from src.evaluation.mask_stats import (
    mask_area,
    find_slices_with_label,
    find_slices_with_nonzero_mask,
)
from src.evaluation.dice import dice_score
from src.models.baseline import segment_liver_hu_threshold_slice
from src.models.medsam_lite import (
    MedSAMLiteUnavailableError,
    get_runtime_status,
    predict_slice_with_bbox,
)
from src.prompting.bbox_auto import get_bbox_from_mask
from src.preprocessing.windowing import apply_liver_window
from src.utils.config import load_config
from src.visualization.overlay import (
    MASK_NONZERO_MODE,
    MASK_TARGET_LABEL_MODE,
    create_binary_mask_display,
    create_mask_overlay,
    draw_bbox_on_image,
    normalize_slice_for_display,
)


RAW_DISPLAY_MODE = "Raw CT slice"
LIVER_WINDOW_DISPLAY_MODE = "Liver windowed CT slice"
SLICE_STATE_KEY = "axial_slice_index"
BBOX_SOURCE_GT = "Ground-truth mask bbox (debug only)"
BBOX_SOURCE_BASELINE = "HU baseline bbox"


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


def _rgb_from_grayscale(image_slice: np.ndarray) -> np.ndarray:
    normalized = normalize_slice_for_display(image_slice)
    return np.stack([normalized, normalized, normalized], axis=-1)


def _render_baseline_and_bbox(
    image_slice: np.ndarray,
    display_slice: np.ndarray,
    mask_display: np.ndarray | None,
    show_baseline: bool,
    show_bbox: bool,
    bbox_source: str,
) -> None:
    baseline_mask = segment_liver_hu_threshold_slice(image_slice)
    gt_area = mask_area(mask_display) if mask_display is not None else None
    baseline_area = mask_area(baseline_mask)
    baseline_overlay = create_mask_overlay(
        image_slice=display_slice,
        mask_slice=baseline_mask,
        alpha=0.4,
        label=None,
    )

    selected_bbox_mask = baseline_mask
    if bbox_source == BBOX_SOURCE_GT and mask_display is not None:
        selected_bbox_mask = mask_display
    bbox = get_bbox_from_mask(selected_bbox_mask)
    bbox_image = draw_bbox_on_image(_rgb_from_grayscale(display_slice), bbox)

    with st.expander("HU baseline and bounding box", expanded=True):
        st.warning(
            "This is a simple HU-threshold heuristic baseline, not a clinical "
            "segmentation model."
        )
        if bbox_source == BBOX_SOURCE_GT:
            st.info(
                "Ground-truth bbox is for debug/control only. The future automatic "
                "MedSAM bbox should come from the HU baseline or another prediction."
            )

        metrics = {
            "bbox_source": bbox_source,
            "bbox_coordinates": bbox,
            "baseline_mask_area": baseline_area,
            "ground_truth_mask_area": gt_area,
        }
        if mask_display is not None:
            metrics["baseline_vs_gt_dice_current_slice"] = dice_score(
                baseline_mask,
                mask_display,
                label=1,
            )
        st.write(metrics)

    columns = st.columns(3)
    if show_baseline:
        with columns[0]:
            st.image(baseline_mask.astype(float), caption="HU baseline mask", clamp=True)
        with columns[1]:
            st.image(baseline_overlay, caption="HU baseline overlay", clamp=True)
    if show_bbox:
        with columns[2]:
            st.image(bbox_image, caption="Bounding box", clamp=True)


def _render_medsam_lite_section(
    image_slice: np.ndarray,
    display_slice: np.ndarray,
    baseline_bbox: tuple[int, int, int, int] | None,
    gt_bbox: tuple[int, int, int, int] | None,
    gt_mask_display: np.ndarray | None,
) -> None:
    config = load_config()
    medsam_config = config.get("medsam", {})
    checkpoint_path = Path(
        medsam_config.get("checkpoint_path", "models/medsam_lite/medsam_lite.pth")
    )
    configured_device = str(medsam_config.get("device", "auto"))
    default_enabled = bool(medsam_config.get("enabled", False))

    with st.expander("MedSAM Lite", expanded=False):
        st.warning(
            "MedSAM Lite inference requires local weights. No model weights are "
            "included in this repository. No clinical use."
        )
        enabled = st.checkbox("Enable MedSAM Lite", value=default_enabled)

        runtime_status = get_runtime_status(checkpoint_path, configured_device)

        st.write(
            {
                "checkpoint_path": runtime_status["checkpoint_path"],
                "checkpoint_status": (
                    "found" if runtime_status["checkpoint_found"] else "missing"
                ),
                "torch_status": (
                    "available" if runtime_status["torch_available"] else "unavailable"
                ),
                "medsam_api_status": (
                    "available"
                    if runtime_status["medsam_api_available"]
                    else "unavailable"
                ),
                "medsam_api_modules": runtime_status["medsam_api_modules"],
                "device_selected": runtime_status["device_selected"],
            }
        )

        medsam_bbox_options = [BBOX_SOURCE_BASELINE]
        if gt_bbox is not None:
            medsam_bbox_options.append(BBOX_SOURCE_GT)
        medsam_bbox_source = st.selectbox(
            "MedSAM Lite bbox prompt source",
            options=medsam_bbox_options,
            help="Ground-truth bbox is for debugging only.",
        )
        if medsam_bbox_source == BBOX_SOURCE_GT:
            st.info(
                "Ground-truth bbox is for debugging only and must not be reported "
                "as an automatic result."
            )

        selected_bbox = (
            gt_bbox if medsam_bbox_source == BBOX_SOURCE_GT else baseline_bbox
        )
        st.write({"selected_bbox": selected_bbox})

        if not enabled:
            st.info("Enable MedSAM Lite to check local inference readiness.")
            return

        if st.button("Run MedSAM Lite on current slice"):
            try:
                start_time = time.perf_counter()
                prediction = predict_slice_with_bbox(
                    image_slice=image_slice,
                    bbox=selected_bbox,
                    checkpoint_path=checkpoint_path,
                    device=configured_device,
                )
                inference_time_seconds = time.perf_counter() - start_time
            except (ValueError, MedSAMLiteUnavailableError) as exc:
                st.warning(str(exc))
                return

            if prediction.shape != image_slice.shape:
                st.warning(
                    "MedSAM Lite returned an invalid prediction shape. "
                    f"Expected {image_slice.shape}, got {prediction.shape}."
                )
                return

            prediction_mask = (prediction > 0).astype(np.float32)
            prediction_overlay = create_mask_overlay(
                image_slice=display_slice,
                mask_slice=prediction_mask,
                alpha=0.4,
                label=None,
            )

            metrics = {
                "bbox_used": selected_bbox,
                "bbox_source": medsam_bbox_source,
                "inference_time_seconds": inference_time_seconds,
            }
            if gt_mask_display is not None:
                metrics["medsam_vs_gt_dice_current_slice"] = dice_score(
                    prediction_mask,
                    gt_mask_display,
                    label=1,
                )
            st.write(metrics)

            prediction_col, overlay_col = st.columns(2)
            with prediction_col:
                st.image(
                    prediction_mask,
                    caption="MedSAM Lite prediction",
                    clamp=True,
                )
            with overlay_col:
                st.image(
                    prediction_overlay,
                    caption="MedSAM Lite overlay",
                    clamp=True,
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

    show_baseline = st.checkbox("Show HU baseline segmentation", value=False)
    show_bbox = st.checkbox("Show automatic bounding box", value=False)
    bbox_options = [BBOX_SOURCE_BASELINE]
    if mask_volume is not None:
        bbox_options.append(BBOX_SOURCE_GT)
    bbox_source = st.selectbox(
        "Bounding box source",
        options=bbox_options,
        help=(
            "Use HU baseline bbox as the automatic future MedSAM prompt. "
            "Ground-truth bbox is debug/control only."
        ),
    )

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
        baseline_mask = segment_liver_hu_threshold_slice(image_slice)
        baseline_bbox = get_bbox_from_mask(baseline_mask)
        if show_baseline or show_bbox:
            _render_baseline_and_bbox(
                image_slice=image_slice,
                display_slice=display_slice,
                mask_display=None,
                show_baseline=show_baseline,
                show_bbox=show_bbox,
                bbox_source=bbox_source,
            )
        _render_medsam_lite_section(
            image_slice=image_slice,
            display_slice=display_slice,
            baseline_bbox=baseline_bbox,
            gt_bbox=None,
            gt_mask_display=None,
        )
        st.info("Upload a matching mask to display mask and overlay views.")
        return

    mask_slice = mask_volume[:, :, slice_index]
    mask_display = create_binary_mask_display(
        mask_slice,
        target_label=target_label,
        mask_display_mode=mask_display_mode,
    )
    baseline_mask = segment_liver_hu_threshold_slice(image_slice)
    baseline_bbox = get_bbox_from_mask(baseline_mask)
    gt_bbox = get_bbox_from_mask(mask_display)
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

    if show_baseline or show_bbox:
        _render_baseline_and_bbox(
            image_slice=image_slice,
            display_slice=display_slice,
            mask_display=mask_display,
            show_baseline=show_baseline,
            show_bbox=show_bbox,
            bbox_source=bbox_source,
        )

    _render_medsam_lite_section(
        image_slice=image_slice,
        display_slice=display_slice,
        baseline_bbox=baseline_bbox,
        gt_bbox=gt_bbox,
        gt_mask_display=mask_display,
    )

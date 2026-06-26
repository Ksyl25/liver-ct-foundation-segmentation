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
from src.evaluation.metrics import (
    compute_bbox_coverage,
    compute_iou,
    compute_slice_metrics,
    export_metrics_to_csv,
)
from src.models.baseline import segment_liver_hu_threshold_slice
from src.models.medsam_lite import (
    MedSAMLitePrediction,
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
MEDSAM_PREDICTION_STATE_KEY = "medsam_prediction_current_slice"
MEDSAM_COMPARISON_STATE_KEY = "medsam_bbox_comparison_current_slice"
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


def _prediction_result_to_mask(
    prediction: MedSAMLitePrediction | np.ndarray,
    measured_time_seconds: float,
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray, float]:
    if isinstance(prediction, MedSAMLitePrediction):
        prediction_array = prediction.mask
        inference_time_seconds = prediction.inference_time_seconds
    else:
        prediction_array = np.asarray(prediction)
        inference_time_seconds = measured_time_seconds

    if prediction_array.shape != expected_shape:
        raise ValueError(
            "MedSAM Lite returned an invalid prediction shape. "
            f"Expected {expected_shape}, got {prediction_array.shape}."
        )

    prediction_mask = (prediction_array > 0).astype(np.float32)
    return prediction_mask, inference_time_seconds


def _run_medsam_prediction_for_bbox(
    image_slice: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
    checkpoint_path: Path,
    configured_device: str,
) -> tuple[np.ndarray, float]:
    start_time = time.perf_counter()
    prediction = predict_slice_with_bbox(
        image_slice=image_slice,
        bbox=bbox,
        checkpoint_path=checkpoint_path,
        device=configured_device,
    )
    measured_time_seconds = time.perf_counter() - start_time
    return _prediction_result_to_mask(
        prediction,
        measured_time_seconds,
        image_slice.shape,
    )


def _format_bbox(bbox: tuple[int, int, int, int] | None) -> str:
    if bbox is None:
        return "None"
    return f"({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})"


def _medsam_comparison_row(
    bbox_source: str,
    bbox: tuple[int, int, int, int] | None,
    prediction_mask: np.ndarray | None,
    gt_mask_display: np.ndarray | None,
    inference_time: float | None,
    notes: str,
) -> dict:
    dice_value = None
    iou_value = None
    if prediction_mask is not None and gt_mask_display is not None:
        dice_value = dice_score(prediction_mask, gt_mask_display, label=1)
        iou_value = compute_iou(prediction_mask, gt_mask_display, label=1)

    return {
        "bbox source": bbox_source,
        "bbox coordinates": _format_bbox(bbox),
        "Dice vs GT": dice_value,
        "IoU vs GT": iou_value,
        "inference time": inference_time,
        "notes": notes,
    }


def _default_medsam_comparison_rows(
    baseline_bbox: tuple[int, int, int, int] | None,
    gt_bbox: tuple[int, int, int, int] | None,
) -> list[dict]:
    rows = [
        _medsam_comparison_row(
            bbox_source="HU baseline bbox",
            bbox=baseline_bbox,
            prediction_mask=None,
            gt_mask_display=None,
            inference_time=None,
            notes="Automatic prompt. Run comparison to compute MedSAM metrics.",
        )
    ]
    if gt_bbox is not None:
        rows.append(
            _medsam_comparison_row(
                bbox_source="Ground-truth bbox",
                bbox=gt_bbox,
                prediction_mask=None,
                gt_mask_display=None,
                inference_time=None,
                notes=(
                    "Debug only, not reportable as an automatic result. "
                    "Run comparison to compute MedSAM metrics."
                ),
            )
        )
    return rows


def _comparison_state_matches(
    state: dict | None,
    slice_index: int,
    slice_shape: tuple[int, int],
) -> bool:
    return bool(
        state
        and state.get("slice_index") == slice_index
        and tuple(state.get("slice_shape", ())) == tuple(slice_shape)
    )


def _render_medsam_bbox_comparison(
    enabled: bool,
    slice_index: int,
    image_slice: np.ndarray,
    display_slice: np.ndarray,
    baseline_bbox: tuple[int, int, int, int] | None,
    gt_bbox: tuple[int, int, int, int] | None,
    gt_mask_display: np.ndarray | None,
    checkpoint_path: Path,
    configured_device: str,
) -> None:
    st.subheader("MedSAM bbox comparison")
    st.caption("HU baseline bbox = automatic prompt")
    st.caption(
        "Ground-truth bbox = debug only, not reportable as automatic result"
    )
    st.caption("Comparison runs at most two single-slice inferences. No volume inference.")

    bbox_base_image = _rgb_from_grayscale(display_slice)
    bbox_columns = st.columns(2)
    with bbox_columns[0]:
        baseline_bbox_image = draw_bbox_on_image(
            bbox_base_image,
            baseline_bbox,
            color=(0.0, 1.0, 0.0),
        )
        st.image(
            baseline_bbox_image,
            caption="HU baseline bbox (automatic prompt)",
            clamp=True,
        )
    with bbox_columns[1]:
        if gt_bbox is None:
            st.info("Ground-truth bbox is unavailable on this slice.")
        else:
            gt_bbox_image = draw_bbox_on_image(
                bbox_base_image,
                gt_bbox,
                color=(1.0, 1.0, 0.0),
            )
            st.image(
                gt_bbox_image,
                caption="Ground-truth bbox (debug only)",
                clamp=True,
            )

    can_compare = enabled and gt_bbox is not None and gt_mask_display is not None
    if not enabled:
        st.info("Enable MedSAM Lite to run the bbox comparison.")
    elif gt_bbox is None or gt_mask_display is None:
        st.info("Upload a ground-truth mask with pixels on this slice to compare bbox sources.")

    if can_compare and st.button("Run MedSAM bbox comparison on current slice"):
        rows = []
        comparison_specs = [
            (
                "HU baseline bbox",
                baseline_bbox,
                "Automatic prompt.",
            ),
            (
                "Ground-truth bbox",
                gt_bbox,
                "Debug only, not reportable as an automatic result.",
            ),
        ]
        for source, bbox, notes in comparison_specs:
            try:
                prediction_mask, inference_time = _run_medsam_prediction_for_bbox(
                    image_slice=image_slice,
                    bbox=bbox,
                    checkpoint_path=checkpoint_path,
                    configured_device=configured_device,
                )
                rows.append(
                    _medsam_comparison_row(
                        bbox_source=source,
                        bbox=bbox,
                        prediction_mask=prediction_mask,
                        gt_mask_display=gt_mask_display,
                        inference_time=inference_time,
                        notes=notes,
                    )
                )
            except (ValueError, MedSAMLiteUnavailableError) as exc:
                st.warning(f"{source}: {exc}")
                rows.append(
                    _medsam_comparison_row(
                        bbox_source=source,
                        bbox=bbox,
                        prediction_mask=None,
                        gt_mask_display=gt_mask_display,
                        inference_time=None,
                        notes=f"{notes} Error: {exc}",
                    )
                )

        st.session_state[MEDSAM_COMPARISON_STATE_KEY] = {
            "slice_index": slice_index,
            "slice_shape": tuple(image_slice.shape),
            "rows": rows,
        }

    comparison_state = st.session_state.get(MEDSAM_COMPARISON_STATE_KEY)
    if _comparison_state_matches(comparison_state, slice_index, image_slice.shape):
        comparison_rows = comparison_state["rows"]
    else:
        comparison_rows = _default_medsam_comparison_rows(baseline_bbox, gt_bbox)
    st.dataframe(comparison_rows, use_container_width=True)


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
    slice_index: int,
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
                "cuda_status": (
                    "available" if runtime_status["cuda_available"] else "unavailable"
                ),
                "gpu_name": runtime_status["gpu_name"],
                "medsam_api_status": (
                    "available"
                    if runtime_status["medsam_api_available"]
                    else "unavailable"
                ),
                "medsam_api_modules": runtime_status["medsam_api_modules"],
                "device_selected": runtime_status["device_selected"],
            }
        )
        if runtime_status["single_slice_warning"]:
            st.warning(runtime_status["single_slice_warning"])

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

        _render_medsam_bbox_comparison(
            enabled=enabled,
            slice_index=slice_index,
            image_slice=image_slice,
            display_slice=display_slice,
            baseline_bbox=baseline_bbox,
            gt_bbox=gt_bbox,
            gt_mask_display=gt_mask_display,
            checkpoint_path=checkpoint_path,
            configured_device=configured_device,
        )

        if not enabled:
            st.info("Enable MedSAM Lite to check local inference readiness.")
            return

        if st.button("Run MedSAM Lite on current slice"):
            try:
                prediction_mask, inference_time_seconds = _run_medsam_prediction_for_bbox(
                    image_slice=image_slice,
                    bbox=selected_bbox,
                    checkpoint_path=checkpoint_path,
                    configured_device=configured_device,
                )
            except (ValueError, MedSAMLiteUnavailableError) as exc:
                st.warning(str(exc))
                return
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
                metrics["medsam_vs_gt_iou_current_slice"] = compute_iou(
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

            st.session_state[MEDSAM_PREDICTION_STATE_KEY] = {
                "slice_index": slice_index,
                "slice_shape": tuple(image_slice.shape),
                "prediction": prediction_mask,
                "bbox": selected_bbox,
                "bbox_source": medsam_bbox_source,
                "inference_time": inference_time_seconds,
            }


def _metric_row(
    method: str,
    slice_index: int,
    target: str,
    metrics: dict,
    bbox: tuple[int, int, int, int] | None,
    bbox_source: str | None,
    bbox_coverage: float | None,
    inference_time: float | None = None,
    notes: str = "",
) -> dict:
    return {
        "case_id": "unknown",
        "slice_index": slice_index,
        "method": method,
        "target": target,
        "dice": metrics.get("dice"),
        "iou": metrics.get("iou"),
        "mask_area": metrics.get("mask_area"),
        "gt_area": metrics.get("gt_area"),
        "bbox": bbox,
        "bbox_source": bbox_source,
        "bbox_coverage": bbox_coverage,
        "inference_time": inference_time,
        "status": metrics.get("status", "available"),
        "notes": notes,
    }


def _render_evaluation_section(
    slice_index: int,
    mask_display: np.ndarray | None,
    baseline_mask: np.ndarray,
    baseline_bbox: tuple[int, int, int, int] | None,
    gt_bbox: tuple[int, int, int, int] | None,
) -> None:
    with st.expander("Evaluation", expanded=True):
        st.warning(
            "Metrics are computed for educational evaluation only and are not "
            "clinically validated."
        )
        if mask_display is None:
            st.info("Upload a ground-truth mask to compute baseline vs GT metrics.")
            return

        rows = []
        gt_metrics = compute_slice_metrics(mask_display, mask_display, label=1)
        rows.append(
            _metric_row(
                method="Ground-truth self-check",
                slice_index=slice_index,
                target="current_slice",
                metrics=gt_metrics,
                bbox=gt_bbox,
                bbox_source=BBOX_SOURCE_GT,
                bbox_coverage=compute_bbox_coverage(mask_display, gt_bbox),
                notes="Ground truth compared with itself.",
            )
        )

        baseline_metrics = compute_slice_metrics(baseline_mask, mask_display, label=1)
        rows.append(
            _metric_row(
                method="HU baseline",
                slice_index=slice_index,
                target="current_slice",
                metrics=baseline_metrics,
                bbox=baseline_bbox,
                bbox_source=BBOX_SOURCE_BASELINE,
                bbox_coverage=compute_bbox_coverage(mask_display, baseline_bbox),
                notes="Naive HU-threshold baseline, not clinical segmentation.",
            )
        )

        medsam_state = st.session_state.get(MEDSAM_PREDICTION_STATE_KEY)
        if (
            medsam_state
            and medsam_state.get("slice_index") == slice_index
            and tuple(medsam_state.get("slice_shape", ())) == tuple(mask_display.shape)
            and medsam_state.get("prediction") is not None
        ):
            medsam_prediction = medsam_state["prediction"]
            medsam_metrics = compute_slice_metrics(
                medsam_prediction,
                mask_display,
                label=1,
            )
            rows.append(
                _metric_row(
                    method="MedSAM Lite",
                    slice_index=slice_index,
                    target="current_slice",
                    metrics=medsam_metrics,
                    bbox=medsam_state.get("bbox"),
                    bbox_source=medsam_state.get("bbox_source"),
                    bbox_coverage=compute_bbox_coverage(
                        mask_display,
                        medsam_state.get("bbox"),
                    ),
                    inference_time=medsam_state.get("inference_time"),
                    notes="Real MedSAM Lite prediction available for this session.",
                )
            )
        else:
            rows.append(
                _metric_row(
                    method="MedSAM Lite",
                    slice_index=slice_index,
                    target="current_slice",
                    metrics={"status": "not_available"},
                    bbox=None,
                    bbox_source=None,
                    bbox_coverage=None,
                    notes="No real MedSAM Lite prediction exists.",
                )
            )

        st.dataframe(rows, use_container_width=True)

        if st.button("Export current metrics to CSV"):
            output_path = export_metrics_to_csv(
                rows,
                Path("outputs") / "metrics" / "metrics_current_slice.csv",
            )
            st.success(f"Metrics exported to {output_path}")


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
            slice_index=slice_index,
            image_slice=image_slice,
            display_slice=display_slice,
            baseline_bbox=baseline_bbox,
            gt_bbox=None,
            gt_mask_display=None,
        )
        _render_evaluation_section(
            slice_index=slice_index,
            mask_display=None,
            baseline_mask=baseline_mask,
            baseline_bbox=baseline_bbox,
            gt_bbox=None,
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
        slice_index=slice_index,
        image_slice=image_slice,
        display_slice=display_slice,
        baseline_bbox=baseline_bbox,
        gt_bbox=gt_bbox,
        gt_mask_display=mask_display,
    )

    _render_evaluation_section(
        slice_index=slice_index,
        mask_display=mask_display,
        baseline_mask=baseline_mask,
        baseline_bbox=baseline_bbox,
        gt_bbox=gt_bbox,
    )

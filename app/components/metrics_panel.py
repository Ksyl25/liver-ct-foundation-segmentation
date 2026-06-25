"""Metadata and metrics panels for the Streamlit MVP."""

from __future__ import annotations

import numpy as np
import streamlit as st

from src.evaluation.dice import dice_score


def render_metadata(metadata: dict, title: str) -> None:
    """Display key NIfTI metadata."""

    st.subheader(title)
    visible_metadata = {
        "path": metadata.get("path"),
        "shape": metadata.get("shape"),
        "dtype": metadata.get("dtype"),
        "spacing": metadata.get("spacing"),
        "min": metadata.get("min"),
        "max": metadata.get("max"),
        "mean": metadata.get("mean"),
    }
    st.json(visible_metadata)

    affine = metadata.get("affine")
    if affine is not None:
        st.caption("Affine")
        st.code(np.array2string(np.asarray(affine), precision=4), language="text")


def render_dice_metrics(mask_volume: np.ndarray | None, target_label: int) -> None:
    """Display the Phase 1 Dice self-check."""

    st.subheader("Dice metrics")
    if mask_volume is None:
        st.info("Upload a mask to compute the Dice self-check.")
        return

    score = dice_score(mask_volume, mask_volume, label=target_label)
    st.metric("Ground-truth self-check Dice", f"{score:.4f}")
    st.caption(
        "Ground-truth self-check Dice. Model prediction will be added in a later phase."
    )


def render_hu_statistics(image_volume: np.ndarray | None) -> None:
    """Display simple HU intensity statistics for the uploaded CT volume."""

    st.subheader("HU statistics")
    if image_volume is None:
        st.info("Upload a CT image to display HU statistics.")
        return

    values = np.asarray(image_volume, dtype=np.float32)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        st.warning("No finite HU values found in the image volume.")
        return

    stats = {
        "min": float(np.min(finite_values)),
        "max": float(np.max(finite_values)),
        "mean": float(np.mean(finite_values)),
        "percentile_1": float(np.percentile(finite_values, 1)),
        "percentile_99": float(np.percentile(finite_values, 99)),
    }
    st.json(stats)

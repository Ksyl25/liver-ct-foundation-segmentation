"""Safe MedSAM Lite integration scaffold.

Phase 4A prepares validation, image formatting and local checkpoint handling
without shipping weights, downloading models or pretending that inference is
available when the local MedSAM Lite API is not installed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from src.preprocessing.normalization import normalize_to_range

BBox = tuple[int, int, int, int]
MEDSAM_API_CANDIDATES = (
    "medsam_lite",
    "MedSAM_Lite",
    "medsam",
    "segment_anything",
)


class MedSAMLiteUnavailableError(RuntimeError):
    """Raised when MedSAM Lite inference cannot run in the local environment."""


def detect_medsam_lite_api() -> dict[str, Any]:
    """Detect local MedSAM/SAM-like Python APIs without importing heavy modules."""

    modules = {
        module_name: importlib.util.find_spec(module_name) is not None
        for module_name in MEDSAM_API_CANDIDATES
    }
    available_modules = [
        module_name for module_name, is_available in modules.items() if is_available
    ]
    return {
        "available": bool(available_modules),
        "available_modules": available_modules,
        "checked_modules": modules,
    }


def get_runtime_status(
    checkpoint_path: str | Path,
    device: str = "auto",
) -> dict[str, Any]:
    """Return local MedSAM Lite readiness information for UI diagnostics."""

    checkpoint = Path(checkpoint_path)
    torch_available = importlib.util.find_spec("torch") is not None
    api_status = detect_medsam_lite_api()

    try:
        selected_device = get_device(device)
    except MedSAMLiteUnavailableError as exc:
        selected_device = f"unavailable: {exc}"

    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_found": checkpoint.exists() and checkpoint.is_file(),
        "torch_available": torch_available,
        "medsam_api_available": api_status["available"],
        "medsam_api_modules": api_status["available_modules"],
        "device_selected": selected_device,
    }


def validate_bbox(
    bbox: tuple[int, int, int, int] | list[int] | None,
    image_shape: tuple[int, ...],
) -> BBox:
    """Validate and return a bbox as integer (x_min, y_min, x_max, y_max)."""

    if bbox is None:
        raise ValueError("bbox must not be None.")
    if len(bbox) != 4:
        raise ValueError("bbox must contain exactly four values.")
    if len(image_shape) < 2:
        raise ValueError("image_shape must contain at least height and width.")

    height, width = int(image_shape[0]), int(image_shape[1])
    x_min, y_min, x_max, y_max = (int(value) for value in bbox)

    if x_max <= x_min or y_max <= y_min:
        raise ValueError("bbox must satisfy x_max > x_min and y_max > y_min.")
    if x_min < 0 or y_min < 0 or x_max >= width or y_max >= height:
        raise ValueError(
            f"bbox {bbox} is outside image bounds width={width}, height={height}."
        )

    return x_min, y_min, x_max, y_max


def prepare_slice_for_medsam(image_slice: np.ndarray) -> np.ndarray:
    """Normalize a 2D CT slice and convert it to pseudo-RGB HxWx3."""

    image = np.asarray(image_slice, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image slice, got shape {image.shape}.")

    normalized = normalize_to_range(image, output_min=0.0, output_max=1.0)
    return np.stack([normalized, normalized, normalized], axis=-1).astype(np.float32)


def get_device(device: str = "auto") -> str:
    """Return cuda when requested and available, otherwise cpu."""

    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda.")
    if device == "cpu":
        return "cpu"

    try:
        import torch
    except ImportError:
        if device == "cuda":
            raise MedSAMLiteUnavailableError(
                "Torch is not installed, so CUDA cannot be selected."
            ) from None
        return "cpu"

    cuda_available = bool(torch.cuda.is_available())
    if device == "cuda" and not cuda_available:
        raise MedSAMLiteUnavailableError("CUDA was requested but is not available.")
    return "cuda" if cuda_available and device == "auto" else device


def load_medsam_lite_model(
    checkpoint_path: str | Path,
    device: str = "auto",
) -> Any:
    """Validate local MedSAM Lite requirements and load a future model."""

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise MedSAMLiteUnavailableError(
            "MedSAM Lite checkpoint not found. Place local weights in "
            "models/medsam_lite/ and configure the path."
        )
    if not checkpoint.is_file():
        raise MedSAMLiteUnavailableError(
            f"MedSAM Lite checkpoint path is not a file: {checkpoint}"
        )

    selected_device = get_device(device)
    try:
        import torch  # noqa: F401
    except ImportError:
        raise MedSAMLiteUnavailableError(
            "Torch is required for MedSAM Lite inference but is not installed."
        ) from None

    api_status = detect_medsam_lite_api()
    if not api_status["available"]:
        checked = ", ".join(api_status["checked_modules"].keys())
        raise MedSAMLiteUnavailableError(
            "MedSAM Lite API is not available locally. Install or expose a local "
            f"MedSAM Lite implementation before running inference. Checked: {checked}."
        )

    raise MedSAMLiteUnavailableError(
        "Real MedSAM Lite model loading is not implemented yet. "
        f"Checkpoint was found and device '{selected_device}' was selected, "
        "but no supported local MedSAM Lite loader has been mapped yet. "
        "No fake model was loaded."
    )


def predict_slice_with_bbox(
    image_slice: np.ndarray,
    bbox: tuple[int, int, int, int] | list[int] | None,
    model: Any | None = None,
    checkpoint_path: str | Path | None = None,
    device: str = "auto",
) -> np.ndarray:
    """Run future MedSAM Lite inference for one 2D slice and bbox prompt."""

    image = np.asarray(image_slice)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image slice, got shape {image.shape}.")

    clean_bbox = validate_bbox(bbox, image.shape)
    _ = prepare_slice_for_medsam(image)
    _ = clean_bbox

    if model is None:
        if checkpoint_path is None:
            raise MedSAMLiteUnavailableError(
                "MedSAM Lite model or checkpoint_path is required for inference."
            )
        model = load_medsam_lite_model(checkpoint_path, device=device)

    raise MedSAMLiteUnavailableError(
        "Real MedSAM Lite inference is not implemented yet. "
        "No fake prediction was produced."
    )

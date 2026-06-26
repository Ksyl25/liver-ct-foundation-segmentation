"""Safe MedSAM Lite integration scaffold.

Phase 4A prepares validation, image formatting and local checkpoint handling
without shipping weights, downloading models or pretending that inference is
available when the local MedSAM Lite API is not installed.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.preprocessing.normalization import normalize_to_range

BBox = tuple[int, int, int, int]
MEDSAM_API_CANDIDATES = (
    "external/MedSAM-Lite",
    "medsam_lite",
    "MedSAM_Lite",
    "medsam",
    "segment_anything",
)
EXTERNAL_LITEMEDSAM_PATH = Path("external") / "MedSAM-Lite"


class MedSAMLiteUnavailableError(RuntimeError):
    """Raised when MedSAM Lite inference cannot run in the local environment."""


@dataclass
class MedSAMLiteLocalModel:
    """Container for a locally loaded LiteMedSAM model and runtime details."""

    model: Any
    device: str
    api_name: str


@dataclass
class MedSAMLitePrediction:
    """A real MedSAM Lite prediction and timing metadata."""

    mask: np.ndarray
    inference_time_seconds: float


def _external_litemedsam_available() -> bool:
    return (EXTERNAL_LITEMEDSAM_PATH / "tiny_vit_sam.py").exists() and (
        EXTERNAL_LITEMEDSAM_PATH / "segment_anything"
    ).is_dir()


def detect_medsam_lite_api() -> dict[str, Any]:
    """Detect local MedSAM/SAM-like Python APIs without importing heavy modules."""

    modules = {
        module_name: (
            _external_litemedsam_available()
            if module_name == "external/MedSAM-Lite"
            else importlib.util.find_spec(module_name) is not None
        )
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
    cuda_available = False
    gpu_name = None
    if torch_available:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                gpu_name = str(torch.cuda.get_device_name(0))
        except Exception:
            cuda_available = False

    try:
        selected_device = get_device(device)
    except MedSAMLiteUnavailableError as exc:
        selected_device = f"unavailable: {exc}"

    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_found": checkpoint.exists() and checkpoint.is_file(),
        "torch_available": torch_available,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "medsam_api_available": api_status["available"],
        "medsam_api_modules": api_status["available_modules"],
        "device_selected": selected_device,
        "single_slice_warning": (
            "NVIDIA GeForce RTX 3050 Laptop GPU detected: run single-slice "
            "inference only to reduce OOM risk."
            if gpu_name and "RTX 3050 Laptop" in gpu_name
            else None
        ),
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


def _ensure_external_litemedsam_import_path() -> Path:
    external_path = EXTERNAL_LITEMEDSAM_PATH.resolve()
    if not _external_litemedsam_available():
        raise MedSAMLiteUnavailableError(
            "external/MedSAM-Lite API files were not found. Place the local "
            "LiteMedSAM source in external/MedSAM-Lite or install a supported API."
        )
    if str(external_path) not in sys.path:
        sys.path.insert(0, str(external_path))
    return external_path


class _LiteMedSAMNetwork:
    """Small adapter around the real LiteMedSAM modules from external/MedSAM-Lite."""

    def __init__(self, torch_module: Any, nn_module: Any, functional_module: Any) -> None:
        class MedSAMLiteModule(nn_module.Module):
            def __init__(self, image_encoder, mask_decoder, prompt_encoder):
                super().__init__()
                self.image_encoder = image_encoder
                self.mask_decoder = mask_decoder
                self.prompt_encoder = prompt_encoder

            @torch_module.no_grad()
            def postprocess_masks(self, masks, new_size, original_size):
                masks = masks[..., : new_size[0], : new_size[1]]
                return functional_module.interpolate(
                    masks,
                    size=(original_size[0], original_size[1]),
                    mode="bilinear",
                    align_corners=False,
                )

        self.module_class = MedSAMLiteModule


def _build_external_litemedsam_model() -> Any:
    _ensure_external_litemedsam_import_path()
    try:
        import torch
        import torch.nn.functional as F
        from tiny_vit_sam import TinyViT
        from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
    except ImportError as exc:
        raise MedSAMLiteUnavailableError(
            "LiteMedSAM dependencies are missing. Install the local LiteMedSAM "
            "requirements, including torch, timm and segment_anything modules."
        ) from exc

    image_encoder = TinyViT(
        img_size=256,
        in_chans=3,
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=0.8,
    )
    prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(256, 256),
        mask_in_chans=16,
    )
    mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=256,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )
    network = _LiteMedSAMNetwork(torch, torch.nn, F).module_class
    return network(image_encoder, mask_decoder, prompt_encoder)


def load_medsam_lite_model(
    checkpoint_path: str | Path,
    device: str = "auto",
) -> Any:
    """Load a real local LiteMedSAM model when source, deps and checkpoint exist."""

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
        import torch
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

    if not _external_litemedsam_available():
        raise MedSAMLiteUnavailableError(
            "A MedSAM-like module was detected, but no supported LiteMedSAM loader "
            "has been mapped for it. No fake model was loaded."
        )

    try:
        model = _build_external_litemedsam_model()
        checkpoint_data = torch.load(str(checkpoint), map_location="cpu")
        state_dict = checkpoint_data.get("model", checkpoint_data) if isinstance(
            checkpoint_data, dict
        ) else checkpoint_data
        model.load_state_dict(state_dict)
        model.to(torch.device(selected_device))
        model.eval()
    except MedSAMLiteUnavailableError:
        raise
    except Exception as exc:
        raise MedSAMLiteUnavailableError(
            "Failed to load the local LiteMedSAM checkpoint. Check that the "
            "checkpoint format matches external/MedSAM-Lite."
        ) from exc

    return MedSAMLiteLocalModel(
        model=model,
        device=selected_device,
        api_name="external/MedSAM-Lite",
    )


def _resize_longest_side(image: np.ndarray, target_length: int = 256) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise MedSAMLiteUnavailableError(
            "Pillow is required for LiteMedSAM image resizing."
        ) from exc

    height, width = image.shape[:2]
    scale = target_length / float(max(height, width))
    new_width = int(width * scale + 0.5)
    new_height = int(height * scale + 0.5)
    pil_image = Image.fromarray((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8))
    resized = pil_image.resize((new_width, new_height), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def _pad_image(image: np.ndarray, target_size: int = 256) -> np.ndarray:
    height, width = image.shape[:2]
    pad_height = target_size - height
    pad_width = target_size - width
    if pad_height < 0 or pad_width < 0:
        raise ValueError("image must be resized before padding.")
    return np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)))


def _scale_bbox_to_resized_image(
    bbox: BBox,
    original_size: tuple[int, int],
    resized_size: tuple[int, int],
) -> np.ndarray:
    original_height, original_width = original_size
    resized_height, resized_width = resized_size
    scale_x = resized_width / float(original_width)
    scale_y = resized_height / float(original_height)
    x_min, y_min, x_max, y_max = bbox
    return np.array(
        [
            x_min * scale_x,
            y_min * scale_y,
            x_max * scale_x,
            y_max * scale_y,
        ],
        dtype=np.float32,
    )[None, :]


def _run_real_litemedsam_inference(
    image_rgb: np.ndarray,
    bbox: BBox,
    loaded_model: MedSAMLiteLocalModel,
) -> np.ndarray:
    try:
        import torch
    except ImportError:
        raise MedSAMLiteUnavailableError(
            "Torch is required for MedSAM Lite inference but is not installed."
        ) from None

    original_size = image_rgb.shape[:2]
    resized = _resize_longest_side(image_rgb, target_length=256)
    resized_size = resized.shape[:2]
    padded = _pad_image(resized, target_size=256)
    box_256 = _scale_bbox_to_resized_image(bbox, original_size, resized_size)

    device = torch.device(loaded_model.device)
    image_tensor = (
        torch.from_numpy(padded)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )

    try:
        with torch.inference_mode():
            image_embedding = loaded_model.model.image_encoder(image_tensor)
            box_torch = torch.as_tensor(box_256, dtype=torch.float32, device=device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]
            sparse_embeddings, dense_embeddings = loaded_model.model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
            low_res_logits, _ = loaded_model.model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=loaded_model.model.prompt_encoder.get_dense_pe().to(device),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            masks = loaded_model.model.postprocess_masks(
                low_res_logits,
                resized_size,
                original_size,
            )
            prediction = torch.sigmoid(masks).squeeze().detach().cpu().numpy()
    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" in message or "cuda" in message:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise MedSAMLiteUnavailableError(
                "CUDA error during single-slice MedSAM Lite inference. "
                "GPU memory may be insufficient; try CPU or a smaller input."
            ) from exc
        raise

    return (prediction > 0.5).astype(np.uint8)


def predict_slice_with_bbox(
    image_slice: np.ndarray,
    bbox: tuple[int, int, int, int] | list[int] | None,
    model: Any | None = None,
    checkpoint_path: str | Path | None = None,
    device: str = "auto",
) -> np.ndarray | MedSAMLitePrediction:
    """Run future MedSAM Lite inference for one 2D slice and bbox prompt."""

    image = np.asarray(image_slice)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image slice, got shape {image.shape}.")

    clean_bbox = validate_bbox(bbox, image.shape)
    image_rgb = prepare_slice_for_medsam(image)

    if model is None:
        if checkpoint_path is None:
            raise MedSAMLiteUnavailableError(
                "MedSAM Lite model or checkpoint_path is required for inference."
            )
        model = load_medsam_lite_model(checkpoint_path, device=device)

    if hasattr(model, "predict_slice"):
        prediction = model.predict_slice(image_rgb, clean_bbox)
        prediction = np.asarray(prediction)
        if prediction.shape != image.shape:
            raise ValueError(
                f"Mock/custom MedSAM model returned shape {prediction.shape}, "
                f"expected {image.shape}."
            )
        return (prediction > 0).astype(np.uint8)

    if not isinstance(model, MedSAMLiteLocalModel):
        raise MedSAMLiteUnavailableError(
            "Unsupported MedSAM Lite model object. No fake prediction was produced."
        )

    start_time = time.perf_counter()
    mask = _run_real_litemedsam_inference(image_rgb, clean_bbox, model)
    inference_time_seconds = time.perf_counter() - start_time
    if mask.shape != image.shape:
        raise ValueError(f"MedSAM Lite returned shape {mask.shape}, expected {image.shape}.")
    return MedSAMLitePrediction(mask=mask, inference_time_seconds=inference_time_seconds)

"""Lightweight command line interface for local project workflows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.data.dicom_loader import DicomLoadError, load_dicom_series
from src.data.nifti_loader import load_nifti, validate_nifti_pair
from src.evaluation.dice import dice_score
from src.evaluation.mask_stats import mask_area
from src.evaluation.metrics import compute_iou, export_metrics_to_csv
from src.models.baseline import segment_liver_hu_threshold
from src.utils.config import ConfigLoadError, load_config


def _print_title(title: str) -> None:
    print(f"\n{title}")
    print("=" * len(title))


def _print_mapping(mapping: dict[str, Any]) -> None:
    for key, value in mapping.items():
        print(f"{key}: {value}")


def _metadata_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "shape": metadata.get("shape"),
        "dtype": metadata.get("dtype"),
        "spacing": metadata.get("spacing"),
        "min": metadata.get("min"),
        "max": metadata.get("max"),
        "mean": metadata.get("mean"),
    }


def run_viewer_command(_args: argparse.Namespace) -> int:
    """Print the Streamlit viewer command."""

    _print_title("Run Viewer")
    print("streamlit run app/dashboard.py")
    return 0


def inspect_nifti_command(args: argparse.Namespace) -> int:
    """Inspect a NIfTI image and optional mask."""

    _print_title("Inspect NIfTI")
    image_volume, image_metadata = load_nifti(args.image)
    print(f"image: {args.image}")
    _print_mapping(_metadata_summary(image_metadata))

    if args.mask:
        mask_volume, mask_metadata = load_nifti(args.mask)
        validate_nifti_pair(image_volume, mask_volume)
        unique_values = np.unique(mask_volume)
        print(f"\nmask: {args.mask}")
        _print_mapping(_metadata_summary(mask_metadata))
        print(f"mask_unique_values: {unique_values[:30].tolist()}")
        if unique_values.size > 30:
            print(f"mask_unique_values_total: {unique_values.size}")
        print(f"mask_nonzero_pixels: {int(np.count_nonzero(mask_volume))}")

    return 0


def inspect_dicom_command(args: argparse.Namespace) -> int:
    """Inspect a local DICOM series using safe metadata only."""

    _print_title("Inspect DICOM")
    _volume, metadata = load_dicom_series(args.dicom_dir)
    print(f"dicom_dir: {args.dicom_dir}")
    safe_summary = {
        "shape": metadata.get("shape"),
        "modality": metadata.get("modality"),
        "series_description": metadata.get("series_description"),
        "pixel_spacing": metadata.get("pixel_spacing"),
        "slice_thickness": metadata.get("slice_thickness"),
        "rescale_slope": metadata.get("rescale_slope"),
        "rescale_intercept": metadata.get("rescale_intercept"),
        "number_of_slices": metadata.get("number_of_slices"),
        "patient_identifiers": metadata.get("patient_identifiers"),
    }
    _print_mapping(safe_summary)
    return 0


def evaluate_nifti_command(args: argparse.Namespace) -> int:
    """Evaluate HU baseline against a NIfTI ground-truth mask."""

    _print_title("Evaluate NIfTI")
    image_volume, _image_metadata = load_nifti(args.image)
    mask_volume, _mask_metadata = load_nifti(args.mask)
    validate_nifti_pair(image_volume, mask_volume)

    baseline_mask = segment_liver_hu_threshold(image_volume)
    self_check_dice = dice_score(mask_volume, mask_volume, label=args.label)
    baseline_dice = dice_score(baseline_mask, mask_volume, label=args.label)
    baseline_iou = compute_iou(baseline_mask, mask_volume, label=args.label)
    baseline_area = mask_area(baseline_mask)
    gt_area = int(np.count_nonzero(mask_volume == args.label))

    results = {
        "image": args.image,
        "mask": args.mask,
        "label": args.label,
        "ground_truth_self_check_dice": self_check_dice,
        "hu_baseline_vs_gt_dice": baseline_dice,
        "hu_baseline_vs_gt_iou": baseline_iou,
        "baseline_area": baseline_area,
        "gt_area": gt_area,
    }
    _print_mapping(results)

    if args.output_csv:
        rows = [
            {
                "case_id": "unknown",
                "slice_index": "volume",
                "method": "Ground-truth self-check",
                "target": f"label_{args.label}",
                "dice": self_check_dice,
                "iou": 1.0,
                "mask_area": gt_area,
                "gt_area": gt_area,
                "status": "available",
                "notes": "Ground truth compared with itself.",
            },
            {
                "case_id": "unknown",
                "slice_index": "volume",
                "method": "HU baseline",
                "target": f"label_{args.label}",
                "dice": baseline_dice,
                "iou": baseline_iou,
                "mask_area": baseline_area,
                "gt_area": gt_area,
                "status": "available",
                "notes": "Naive HU-threshold baseline, not clinical segmentation.",
            },
        ]
        output_path = export_metrics_to_csv(rows, args.output_csv)
        print(f"metrics_csv: {output_path}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        prog="liver-ct-cli",
        description="Local CLI for liver CT NIfTI, DICOM and evaluation workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_viewer = subparsers.add_parser("run-viewer", help="Show the Streamlit command.")
    run_viewer.set_defaults(func=run_viewer_command)

    inspect_nifti = subparsers.add_parser("inspect-nifti", help="Inspect NIfTI metadata.")
    inspect_nifti.add_argument("--image", required=True, help="Path to image NIfTI.")
    inspect_nifti.add_argument("--mask", help="Optional path to mask NIfTI.")
    inspect_nifti.set_defaults(func=inspect_nifti_command)

    inspect_dicom = subparsers.add_parser("inspect-dicom", help="Inspect a DICOM folder.")
    inspect_dicom.add_argument("--dicom-dir", required=True, help="Path to DICOM folder.")
    inspect_dicom.set_defaults(func=inspect_dicom_command)

    evaluate_nifti = subparsers.add_parser(
        "evaluate-nifti",
        help="Evaluate HU baseline against a NIfTI mask.",
    )
    evaluate_nifti.add_argument("--image", required=True, help="Path to image NIfTI.")
    evaluate_nifti.add_argument("--mask", required=True, help="Path to mask NIfTI.")
    evaluate_nifti.add_argument("--label", type=int, default=1, help="Target label.")
    evaluate_nifti.add_argument("--output-csv", help="Optional metrics CSV output path.")
    evaluate_nifti.set_defaults(func=evaluate_nifti_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        load_config()
        return int(args.func(args))
    except (ConfigLoadError, DicomLoadError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

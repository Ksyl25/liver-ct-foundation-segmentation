import csv

from src.evaluation.metrics import CSV_COLUMNS, export_metrics_to_csv


def test_export_metrics_to_csv_creates_file_with_expected_columns(tmp_path):
    output_path = tmp_path / "metrics.csv"
    rows = [
        {
            "case_id": "case_001",
            "slice_index": 3,
            "method": "HU baseline",
            "target": "current_slice",
            "dice": 0.5,
            "iou": 0.33,
            "mask_area": 10,
            "gt_area": 12,
            "bbox": (1, 2, 3, 4),
            "bbox_source": "HU baseline bbox",
            "inference_time": None,
            "notes": "test",
        }
    ]

    written_path = export_metrics_to_csv(rows, output_path)

    assert written_path == output_path
    assert output_path.exists()

    with output_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == CSV_COLUMNS
        exported_rows = list(reader)

    assert exported_rows[0]["case_id"] == "case_001"
    assert exported_rows[0]["method"] == "HU baseline"
    assert exported_rows[0]["dice"] == "0.5"

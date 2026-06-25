import pytest

from src.utils.config import ConfigLoadError, get_default_paths, load_config


def test_load_config_returns_empty_dict_when_absent(tmp_path):
    assert load_config(tmp_path / "missing.yaml") == {}


def test_load_config_reads_yaml_mapping(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "data:\n  input_image: image.nii.gz\n  input_mask: mask.nii.gz\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["data"]["input_image"] == "image.nii.gz"


def test_load_config_raises_clear_error_for_invalid_yaml(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("data: [broken\n", encoding="utf-8")

    with pytest.raises(ConfigLoadError, match="Invalid YAML"):
        load_config(config_path)


def test_get_default_paths_returns_configured_paths():
    config = {
        "data": {
            "input_image": "image.nii.gz",
            "input_mask": "mask.nii.gz",
            "dicom_folder": "dicom",
        }
    }

    assert get_default_paths(config) == {
        "input_image": "image.nii.gz",
        "input_mask": "mask.nii.gz",
        "dicom_folder": "dicom",
    }

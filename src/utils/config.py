"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigLoadError(ValueError):
    """Raised when a YAML config file exists but cannot be parsed."""


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load a YAML configuration file, returning an empty dict if absent."""

    config_path = Path(path)
    if not config_path.exists():
        return {}

    try:
        with config_path.open("r", encoding="utf-8") as file:
            loaded_config = yaml.safe_load(file) or {}
    except yaml.YAMLError as exc:
        raise ConfigLoadError(f"Invalid YAML config file: {config_path}") from exc

    if not isinstance(loaded_config, dict):
        raise ConfigLoadError(f"Config file must contain a YAML mapping: {config_path}")

    return loaded_config


def get_default_paths(config: dict[str, Any]) -> dict[str, str]:
    """Extract optional default local paths from a config dictionary."""

    data_config = config.get("data", {}) if isinstance(config, dict) else {}
    return {
        "input_image": str(data_config.get("input_image", "")),
        "input_mask": str(data_config.get("input_mask", "")),
        "dicom_folder": str(data_config.get("dicom_folder", "")),
    }

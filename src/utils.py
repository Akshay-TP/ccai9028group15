from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

"""Project-wide helper utilities."""


REQUIRED_CONFIG_KEYS: dict[str, tuple[str, ...]] = {
    "data": (
        "external_dir",
        "processed_dir",
        "diabetes_zip_url",
        "diabetes_csv_name",
        "hk_stats_file",
    ),
    "model": (
        "target_column",
        "random_state",
        "test_size",
        "calibration_target_rate",
        "positive_threshold",
    ),
    "app": (
        "database_path",
        "model_path",
        "metadata_path",
    ),
}


def _validate_config(config: dict[str, Any], config_path: Path) -> None:
    missing_sections = [section for section in REQUIRED_CONFIG_KEYS if section not in config]
    if missing_sections:
        raise KeyError(
            f"Missing config section(s) {missing_sections} in {config_path}. "
            "Check config/project_config.yaml structure."
        )

    missing_keys: list[str] = []
    for section, keys in REQUIRED_CONFIG_KEYS.items():
        section_value = config.get(section, {})
        if not isinstance(section_value, dict):
            raise TypeError(f"Config section '{section}' must be a mapping in {config_path}.")
        for key in keys:
            if key not in section_value:
                missing_keys.append(f"{section}.{key}")

    if missing_keys:
        raise KeyError(f"Missing config key(s) {missing_keys} in {config_path}.")


def load_config(config_path: str = "config/project_config.yaml") -> dict[str, Any]:
    # Resolve relative config paths from project root, not current working directory.
    project_root = Path(__file__).resolve().parents[1]
    candidate_path = Path(config_path)
    resolved_config_path = candidate_path if candidate_path.is_absolute() else project_root / candidate_path

    # Read one config file so paths and model settings stay centralized.
    with resolved_config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError(f"Config file {resolved_config_path} must contain a top-level mapping.")

    _validate_config(config, resolved_config_path)
    return config

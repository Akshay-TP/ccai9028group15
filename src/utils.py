from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

"""Project-wide helper utilities."""


def load_config(config_path: str = "config/project_config.yaml") -> dict[str, Any]:
    # Resolve relative config paths from project root, not current working directory.
    project_root = Path(__file__).resolve().parents[1]
    candidate_path = Path(config_path)
    resolved_config_path = candidate_path if candidate_path.is_absolute() else project_root / candidate_path

    # Read one config file so paths and model settings stay centralized.
    with resolved_config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)

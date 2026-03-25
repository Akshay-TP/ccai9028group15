from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

"""
Shared utility helpers used across the project.

What this file does:
1. Loads the central YAML configuration.

Possible improvements:
1. Validate required config keys and show clearer errors when keys are missing.
2. Add environment variable overrides for secrets and deployment settings.
"""


def load_config(config_path: str = "config/project_config.yaml") -> dict[str, Any]:
    # Resolve relative config paths from project root, not current working directory.
    project_root = Path(__file__).resolve().parents[1]
    candidate_path = Path(config_path)
    resolved_config_path = candidate_path if candidate_path.is_absolute() else project_root / candidate_path

    # Read one config file so paths and model settings stay centralized.
    with resolved_config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)

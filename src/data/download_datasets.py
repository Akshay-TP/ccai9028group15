from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

if __package__ in (None, ""):
    # Support direct file execution by making project root importable.
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils import load_config

"""Step 1 of the pipeline: download source files into data/external."""


def download_diabetes_dataset(output_dir: Path, zip_url: str) -> Path:
    # Ensure destination exists before writing downloaded content.
    output_dir.mkdir(parents=True, exist_ok=True)
    response = requests.get(zip_url, timeout=60)
    response.raise_for_status()

    # Extract all files directly from the downloaded in-memory zip.
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(output_dir)

    candidate_files = list(output_dir.rglob("*.csv"))
    if not candidate_files:
        raise FileNotFoundError("No CSV files were extracted from the downloaded archive.")

    diabetes_csv = next((f for f in candidate_files if "diabetic_data" in f.name.lower()), None)
    if diabetes_csv is None:
        diabetes_csv = candidate_files[0]

    return diabetes_csv


def build_hk_stats_snapshot(config: dict) -> pd.DataFrame:
    hk_stats_path = Path(config["data"]["hk_stats_file"])
    if not hk_stats_path.exists():
        raise FileNotFoundError(f"Hong Kong stats file missing: {hk_stats_path}")
    return pd.read_csv(hk_stats_path)


def main(config_path: str) -> None:
    config = load_config(config_path)
    external_dir = Path(config["data"]["external_dir"])
    zip_url = config["data"]["diabetes_zip_url"]

    # Download primary public dataset.
    csv_path = download_diabetes_dataset(external_dir, zip_url)
    print(f"Downloaded dataset to: {csv_path}")

    # Save HK stats into run folder to keep all model inputs together.
    hk_stats = build_hk_stats_snapshot(config)
    hk_stats_output = external_dir / "hk_stats_snapshot.csv"
    hk_stats.to_csv(hk_stats_output, index=False)
    print(f"Saved Hong Kong stats snapshot to: {hk_stats_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download EHR-like public dataset and HK statistics snapshot.")
    parser.add_argument("--config", default="config/project_config.yaml", help="Path to project config")
    args = parser.parse_args()
    main(args.config)

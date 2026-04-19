from __future__ import annotations

import argparse
import hashlib
import io
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yaml

if __package__ in (None, ""):
    # Support direct file execution by making project root importable.
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils import load_config

"""Step 1 of the pipeline: download source files into data/external."""


def _resolve_project_path(path_value: str) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    candidate = Path(path_value)
    return candidate if candidate.is_absolute() else project_root / candidate


def _download_with_retry(zip_url: str, retries: int, timeout_seconds: int) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(zip_url, timeout=timeout_seconds)
            response.raise_for_status()
            return response.content
        except requests.RequestException as exc:
            last_error = exc
            if attempt < retries:
                # Exponential backoff helps with transient network and endpoint throttling.
                sleep_seconds = 2 ** (attempt - 1)
                time.sleep(sleep_seconds)

    raise RuntimeError(f"Failed to download dataset after {retries} attempt(s): {last_error}")


def download_diabetes_dataset(
    output_dir: Path,
    zip_url: str,
    retries: int,
    timeout_seconds: int,
    expected_sha256: str | None,
) -> tuple[Path, str, int]:
    # Ensure destination exists before writing downloaded content.
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_bytes = _download_with_retry(zip_url, retries=retries, timeout_seconds=timeout_seconds)
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    if expected_sha256 and archive_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            "Downloaded archive checksum mismatch. "
            f"Expected {expected_sha256}, got {archive_sha256}."
        )

    # Extract all files directly from the downloaded in-memory zip.
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zip_ref:
        zip_ref.extractall(output_dir)

    candidate_files = list(output_dir.rglob("*.csv"))
    if not candidate_files:
        raise FileNotFoundError("No CSV files were extracted from the downloaded archive.")

    diabetes_csv = next((f for f in candidate_files if "diabetic_data" in f.name.lower()), None)
    if diabetes_csv is None:
        diabetes_csv = candidate_files[0]

    return diabetes_csv, archive_sha256, len(archive_bytes)


def build_hk_stats_snapshot(config: dict) -> pd.DataFrame:
    hk_stats_path = _resolve_project_path(config["data"]["hk_stats_file"])
    if not hk_stats_path.exists():
        raise FileNotFoundError(f"Hong Kong stats file missing: {hk_stats_path}")
    return pd.read_csv(hk_stats_path)


def main(config_path: str) -> None:
    config = load_config(config_path)
    external_dir = _resolve_project_path(config["data"]["external_dir"])
    zip_url = config["data"]["diabetes_zip_url"]
    retries = int(config["data"].get("download_retries", 3))
    timeout_seconds = int(config["data"].get("download_timeout_seconds", 60))
    expected_sha256 = config["data"].get("diabetes_zip_sha256")

    # Download primary public dataset.
    csv_path, archive_sha256, archive_size = download_diabetes_dataset(
        output_dir=external_dir,
        zip_url=zip_url,
        retries=retries,
        timeout_seconds=timeout_seconds,
        expected_sha256=expected_sha256,
    )
    print(f"Downloaded dataset to: {csv_path}")

    # Save HK stats into run folder to keep all model inputs together.
    hk_stats = build_hk_stats_snapshot(config)
    hk_stats_output = external_dir / "hk_stats_snapshot.csv"
    hk_stats.to_csv(hk_stats_output, index=False)
    print(f"Saved Hong Kong stats snapshot to: {hk_stats_output}")

    metadata = {
        "source_url": zip_url,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive_size,
        "extracted_csv": str(csv_path),
        "hk_stats_snapshot": str(hk_stats_output),
    }
    metadata_path = external_dir / "download_metadata.yaml"
    with metadata_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(metadata, file, sort_keys=False)
    print(f"Saved download metadata to: {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download EHR-like public dataset and HK statistics snapshot.")
    parser.add_argument("--config", default="config/project_config.yaml", help="Path to project config")
    args = parser.parse_args()
    main(args.config)

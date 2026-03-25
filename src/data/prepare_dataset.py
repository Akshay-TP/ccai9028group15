from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    # Support direct file execution by making project root importable.
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils import load_config

"""
Step 2: Prepare model-ready dataset from raw EHR-like data.

What this file does:
1. Cleans raw fields and handles missing markers.
2. Builds a binary 30-day readmission target.
3. Engineers simple chronic-condition flags from diagnosis code prefixes.
4. Selects a compact feature set for beginner-friendly modeling.

Possible improvements:
1. Add unit tests for feature engineering rules.
2. Use ICD dictionary mapping instead of prefix-only rules.
3. Add data quality reports (missingness, outliers, drift checks).
"""


def _to_numeric(series: pd.Series) -> pd.Series:
    # Convert strings to numeric values and coerce invalid values to NaN.
    return pd.to_numeric(series, errors="coerce")


def _chronic_condition_flags(df: pd.DataFrame) -> pd.DataFrame:
    # Build simple condition flags from diagnosis code prefixes.
    diag_cols = ["diag_1", "diag_2", "diag_3"]
    diag = df[diag_cols].fillna("").astype(str)

    def has_prefix(prefixes: tuple[str, ...]) -> pd.Series:
        return diag.apply(lambda row: any(code.startswith(prefixes) for code in row), axis=1)

    df["flag_diabetes"] = has_prefix(("250",))
    df["flag_heart_failure"] = has_prefix(("428",))
    df["flag_kidney_disease"] = has_prefix(("585",))
    df["flag_copd"] = has_prefix(("491", "492", "496"))
    return df


def _clean_diabetes_data(df: pd.DataFrame) -> pd.DataFrame:
    # The source dataset marks many unknown fields as '?'.
    df = df.replace("?", np.nan).copy()

    # Target label: 1 means readmitted within 30 days.
    df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)

    # Convert age buckets like [60-70) into midpoint numeric value.
    df["age_midpoint"] = (
        df["age"]
        .str.replace("[", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.split("-")
        .apply(lambda x: np.mean([float(v) for v in x]) if isinstance(x, list) and len(x) == 2 else np.nan)
    )

    numeric_cols = [
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
    ]
    for col in numeric_cols:
        df[col] = _to_numeric(df[col])

    # Add comorbidity flags derived from diagnosis fields.
    df = _chronic_condition_flags(df)

    selected_cols = [
        "race",
        "gender",
        "age_midpoint",
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
        "A1Cresult",
        "max_glu_serum",
        "insulin",
        "change",
        "diabetesMed",
        "flag_diabetes",
        "flag_heart_failure",
        "flag_kidney_disease",
        "flag_copd",
        "readmitted_30d",
    ]

    model_df = df[selected_cols].copy()
    # Keep rows with valid target labels.
    model_df = model_df.dropna(subset=["readmitted_30d"])

    return model_df


def main(config_path: str) -> None:
    config = load_config(config_path)
    external_dir = Path(config["data"]["external_dir"])
    processed_dir = Path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Try standard extracted filename first, fallback to first CSV found.
    diabetes_file = next(external_dir.rglob("diabetic_data.csv"), None)
    if diabetes_file is None:
        csv_files = list(external_dir.rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError("No CSV files found in external data directory. Run download script first.")
        diabetes_file = csv_files[0]

    raw_df = pd.read_csv(diabetes_file)
    model_df = _clean_diabetes_data(raw_df)

    output_path = processed_dir / "readmission_model_dataset.csv"
    model_df.to_csv(output_path, index=False)
    print(f"Prepared dataset saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare readmission dataset for model training.")
    parser.add_argument("--config", default="config/project_config.yaml", help="Path to project config")
    args = parser.parse_args()
    main(args.config)

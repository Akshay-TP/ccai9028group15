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

"""Step 2 of the pipeline: clean raw data and build the modeling dataset."""


def _resolve_project_path(path_value: str) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    candidate = Path(path_value)
    return candidate if candidate.is_absolute() else project_root / candidate


def _to_numeric(series: pd.Series) -> pd.Series:
    # Convert strings to numeric values and coerce invalid values to NaN.
    return pd.to_numeric(series, errors="coerce")


def _chronic_condition_flags(df: pd.DataFrame) -> pd.DataFrame:
    # Build simple condition flags from diagnosis code prefixes.
    diag_cols = ["diag_1", "diag_2", "diag_3"]
    diag = df[diag_cols].fillna("").astype(str)

    def has_prefix(prefixes: tuple[str, ...]) -> pd.Series:
        starts_with = pd.DataFrame({col: diag[col].str.startswith(prefixes, na=False) for col in diag_cols})
        return starts_with.any(axis=1)

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

    # Derived utilization and acuity proxies improve risk signal without external labels.
    df["total_prior_visits"] = (
        df["number_outpatient"].fillna(0) + df["number_emergency"].fillna(0) + df["number_inpatient"].fillna(0)
    )
    df["inpatient_visit_ratio"] = df["number_inpatient"].fillna(0) / (df["total_prior_visits"] + 1.0)
    df["medications_per_day"] = df["num_medications"].fillna(0) / (df["time_in_hospital"].fillna(0) + 1.0)
    df["age_over_65"] = (df["age_midpoint"] >= 65).astype(int)

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
        "total_prior_visits",
        "inpatient_visit_ratio",
        "medications_per_day",
        "age_over_65",
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


def _build_quality_report(model_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    total_rows = max(len(model_df), 1)

    for column in model_df.columns:
        series = model_df[column]
        row: dict[str, float | str] = {
            "column": column,
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_ratio": float(series.isna().sum() / total_rows),
            "unique_values": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            row["mean"] = float(series.mean()) if series.notna().any() else np.nan
            row["std"] = float(series.std()) if series.notna().any() else np.nan
            row["min"] = float(series.min()) if series.notna().any() else np.nan
            row["max"] = float(series.max()) if series.notna().any() else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def main(config_path: str) -> None:
    config = load_config(config_path)
    external_dir = _resolve_project_path(config["data"]["external_dir"])
    processed_dir = _resolve_project_path(config["data"]["processed_dir"])
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

    quality_report = _build_quality_report(model_df)
    quality_path = processed_dir / "data_quality_report.csv"
    quality_report.to_csv(quality_path, index=False)
    print(f"Saved data quality report to: {quality_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare readmission dataset for model training.")
    parser.add_argument("--config", default="config/project_config.yaml", help="Path to project config")
    args = parser.parse_args()
    main(args.config)

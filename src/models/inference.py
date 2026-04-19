from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from src.models.calibration import prevalence_shift_calibration

"""Inference wrapper for loading artifacts, scoring patients, and assigning risk labels."""


class ReadmissionScorer:
    def __init__(self, model_path: str, metadata_path: str) -> None:
        # Load model once so repeated dashboard scoring is fast.
        self.model = joblib.load(model_path)
        with Path(metadata_path).open("r", encoding="utf-8") as file:
            self.metadata = yaml.safe_load(file)

        self.features = list(self.metadata.get("features", []))
        self.numeric_features = set(self.metadata.get("numeric_features", []))
        self.categorical_features = set(self.metadata.get("categorical_features", []))
        if not self.features:
            raise ValueError("Model metadata is missing feature definitions.")

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"])

    @property
    def medium_threshold(self) -> float:
        thresholds = self.metadata.get("risk_band_thresholds", {})
        if isinstance(thresholds, dict) and "medium" in thresholds:
            return float(thresholds["medium"])
        return float(max(0.10, self.threshold * 0.65))

    def _validate_and_prepare(self, patient_rows: list[dict[str, Any]]) -> pd.DataFrame:
        if not patient_rows:
            raise ValueError("No patient rows provided for scoring.")

        df = pd.DataFrame(patient_rows)

        # Rebuild engineered features for registry-originating rows that only include base inputs.
        if "total_prior_visits" not in df.columns:
            df["total_prior_visits"] = (
                pd.to_numeric(df.get("number_outpatient", 0), errors="coerce").fillna(0)
                + pd.to_numeric(df.get("number_emergency", 0), errors="coerce").fillna(0)
                + pd.to_numeric(df.get("number_inpatient", 0), errors="coerce").fillna(0)
            )
        if "inpatient_visit_ratio" not in df.columns:
            df["inpatient_visit_ratio"] = (
                pd.to_numeric(df.get("number_inpatient", 0), errors="coerce").fillna(0)
                / (pd.to_numeric(df.get("total_prior_visits", 0), errors="coerce").fillna(0) + 1.0)
            )
        if "medications_per_day" not in df.columns:
            df["medications_per_day"] = (
                pd.to_numeric(df.get("num_medications", 0), errors="coerce").fillna(0)
                / (pd.to_numeric(df.get("time_in_hospital", 0), errors="coerce").fillna(0) + 1.0)
            )
        if "age_over_65" not in df.columns:
            df["age_over_65"] = (pd.to_numeric(df.get("age_midpoint", 0), errors="coerce").fillna(0) >= 65).astype(int)

        missing = [feature for feature in self.features if feature not in df.columns]
        if missing:
            raise KeyError(f"Input rows are missing required feature columns: {missing}")

        # Keep strict feature ordering expected by the training pipeline.
        df = df[self.features].copy()

        for column in self.numeric_features:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

        for column in self.categorical_features:
            if column in df.columns:
                df[column] = df[column].astype("string")

        return df

    def score(self, patient_rows: list[dict[str, Any]]) -> pd.DataFrame:
        # Convert incoming list-of-dicts to dataframe for pipeline compatibility.
        df = self._validate_and_prepare(patient_rows)
        probs = self.model.predict_proba(df)[:, 1]

        # Align probabilities with target deployment prevalence.
        calibrated = prevalence_shift_calibration(
            probabilities=np.array(probs),
            train_prevalence=float(self.metadata["train_prevalence"]),
            target_prevalence=float(self.metadata["target_prevalence"]),
        )
        # Keep legacy binary label and add richer triage band.
        labels = np.where(calibrated >= self.threshold, "HIGH", "LOW")
        bands = np.where(
            calibrated >= self.threshold,
            "HIGH",
            np.where(calibrated >= self.medium_threshold, "MEDIUM", "LOW"),
        )
        output = df.copy()
        output["raw_probability"] = probs
        output["calibrated_probability"] = calibrated
        output["risk_label"] = labels
        output["risk_band"] = bands
        output["scoring_threshold"] = self.threshold
        output["medium_risk_threshold"] = self.medium_threshold
        return output

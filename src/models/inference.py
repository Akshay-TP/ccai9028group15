from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from src.models.calibration import prevalence_shift_calibration

"""
Inference layer used by the dashboard.

What this file does:
1. Loads trained model and metadata.
2. Scores patient records.
3. Applies calibration and risk threshold labeling.

Possible improvements:
1. Add strict schema validation before scoring.
2. Add batch scoring logs for auditability.
3. Add confidence intervals or uncertainty estimates.
"""


class ReadmissionScorer:
    def __init__(self, model_path: str, metadata_path: str) -> None:
        # Load model once so repeated dashboard scoring is fast.
        self.model = joblib.load(model_path)
        with Path(metadata_path).open("r", encoding="utf-8") as file:
            self.metadata = yaml.safe_load(file)

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"])

    def score(self, patient_rows: list[dict[str, Any]]) -> pd.DataFrame:
        # Convert incoming list-of-dicts to dataframe for pipeline compatibility.
        df = pd.DataFrame(patient_rows)
        probs = self.model.predict_proba(df)[:, 1]

        # Align probabilities with target deployment prevalence.
        calibrated = prevalence_shift_calibration(
            probabilities=np.array(probs),
            train_prevalence=float(self.metadata["train_prevalence"]),
            target_prevalence=float(self.metadata["target_prevalence"]),
        )
        # Assign operational labels for triage workflows.
        labels = np.where(calibrated >= self.threshold, "HIGH", "LOW")
        output = df.copy()
        output["raw_probability"] = probs
        output["calibrated_probability"] = calibrated
        output["risk_label"] = labels
        return output

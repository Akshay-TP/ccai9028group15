from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

if __package__ in (None, ""):
    # Support direct file execution by making project root importable.
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.calibration import prevalence_shift_calibration
from src.utils import load_config

"""Step 3 of the pipeline: train candidate models, pick the best, and save artifacts."""


NUMERIC_COLUMNS = [
    "age_midpoint",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

CATEGORICAL_COLUMNS = [
    "race",
    "gender",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "A1Cresult",
    "max_glu_serum",
    "insulin",
    "change",
    "diabetesMed",
    "flag_diabetes",
    "flag_heart_failure",
    "flag_kidney_disease",
    "flag_copd",
]


def build_preprocessor() -> ColumnTransformer:
    # Numeric: fill missing values then standardize scale.
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical: fill missing values then one-hot encode.
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLUMNS),
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
        ]
    )


def evaluate_model(name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    # Evaluate using probability-based metrics for imbalanced classification.
    probs = model.predict_proba(x_test)[:, 1]
    return {
        "model": name,
        "roc_auc": roc_auc_score(y_test, probs),
        "pr_auc": average_precision_score(y_test, probs),
    }


def main(config_path: str) -> None:
    config = load_config(config_path)
    random_state = config["model"]["random_state"]
    target_col = config["model"]["target_column"]
    target_rate = float(config["model"]["calibration_target_rate"])

    dataset_path = Path(config["data"]["processed_dir"]) / "readmission_model_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Processed dataset missing: {dataset_path}")

    # Read model-ready data and split train/test with class stratification.
    df = pd.read_csv(dataset_path)
    x = df.drop(columns=[target_col])
    y = df[target_col]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=config["model"]["test_size"], random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor()

    # Candidate algorithms chosen to show baseline + tree + neural approach.
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "gradient_boosting": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
        ),
        "deep_mlp": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=120, random_state=random_state),
    }

    trained: dict[str, Pipeline] = {}
    metrics: list[dict[str, float]] = []

    for model_name, estimator in models.items():
        # One shared preprocessing pipeline keeps model inputs consistent.
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipe.fit(x_train, y_train)
        trained[model_name] = pipe
        metrics.append(evaluate_model(model_name, pipe, x_test, y_test))

    metrics_df = pd.DataFrame(metrics).sort_values(by="roc_auc", ascending=False)
    best_name = metrics_df.iloc[0]["model"]
    best_model = trained[best_name]

    # Calibrate to target prevalence expected in Hong Kong scenario planning.
    raw_probs = best_model.predict_proba(x_test)[:, 1]
    calibrated_probs = prevalence_shift_calibration(
        probabilities=np.array(raw_probs),
        train_prevalence=float(y_train.mean()),
        target_prevalence=target_rate,
    )

    calibrated_roc_auc = roc_auc_score(y_test, calibrated_probs)
    calibrated_pr_auc = average_precision_score(y_test, calibrated_probs)

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(config["app"]["model_path"])
    metadata_path = Path(config["app"]["metadata_path"])
    metrics_path = artifacts_dir / "model_comparison.csv"

    # Save all outputs needed by dashboard and future reproducibility.
    joblib.dump(best_model, model_path)
    metrics_df.to_csv(metrics_path, index=False)

    metadata = {
        "best_model": str(best_name),
        "train_prevalence": float(y_train.mean()),
        "target_prevalence": target_rate,
        "threshold": float(config["model"]["positive_threshold"]),
        "calibrated_roc_auc": float(calibrated_roc_auc),
        "calibrated_pr_auc": float(calibrated_pr_auc),
        "features": list(x.columns),
        "numeric_features": NUMERIC_COLUMNS,
        "categorical_features": CATEGORICAL_COLUMNS,
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(metadata, file, sort_keys=False)

    print("Training complete.")
    print(f"Best model: {best_name}")
    print(f"Saved model to: {model_path}")
    print(f"Saved metadata to: {metadata_path}")
    print(f"Saved comparison metrics to: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train readmission models and persist best artifact.")
    parser.add_argument("--config", default="config/project_config.yaml", help="Path to project config")
    args = parser.parse_args()
    main(args.config)

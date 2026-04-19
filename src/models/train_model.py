from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ModuleNotFoundError:
    HAS_XGBOOST = False

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
    "total_prior_visits",
    "inpatient_visit_ratio",
    "medications_per_day",
    "age_over_65",
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
    probs = model.predict_proba(x_test)[:, 1]
    return {
        "model": name,
        "roc_auc": roc_auc_score(y_test, probs),
        "pr_auc": average_precision_score(y_test, probs),
    }


def _resolve_project_path(path_value: str) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    candidate = Path(path_value)
    return candidate if candidate.is_absolute() else project_root / candidate


def _candidate_specifications(random_state: int) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {
        "logistic_regression": {
            "estimator": LogisticRegression(max_iter=1600, random_state=random_state),
            "params": {
                "model__C": np.logspace(-2, 1, 10),
                "model__solver": ["lbfgs", "liblinear"],
                "model__class_weight": [None, "balanced"],
            },
            "n_iter": 5,
        },
        "random_forest": {
            "estimator": RandomForestClassifier(
                n_estimators=320,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=1,
            ),
            "params": {
                "model__n_estimators": [240, 320, 420],
                "model__max_depth": [8, 12, 18, None],
                "model__min_samples_split": [2, 5, 12],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", 0.8],
                "model__class_weight": [None, "balanced", "balanced_subsample"],
            },
            "n_iter": 4,
        },
        "deep_mlp": {
            "estimator": MLPClassifier(max_iter=160, random_state=random_state),
            "params": {
                "model__hidden_layer_sizes": [(128, 64), (128, 64, 32), (256, 96)],
                "model__alpha": [1e-4, 5e-4, 1e-3],
                "model__learning_rate_init": [5e-4, 1e-3, 2e-3],
            },
            "n_iter": 5,
        },
    }

    if HAS_XGBOOST:
        specs["gradient_boosting"] = {
            "estimator": XGBClassifier(
                eval_metric="logloss",
                random_state=random_state,
                tree_method="hist",
                n_jobs=1,
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
            ),
            "params": {
                "model__n_estimators": [220, 300, 420],
                "model__learning_rate": [0.03, 0.05, 0.08],
                "model__max_depth": [3, 4, 5],
                "model__min_child_weight": [1, 3, 5],
                "model__subsample": [0.75, 0.9, 1.0],
                "model__colsample_bytree": [0.7, 0.9, 1.0],
            },
            "n_iter": 5,
        }

    return specs


def _classification_metrics(y_true: pd.Series, calibrated_probs: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (calibrated_probs >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, calibrated_probs)),
        "pr_auc": float(average_precision_score(y_true, calibrated_probs)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
    }


def _optimize_threshold(y_true: pd.Series, calibrated_probs: np.ndarray, beta: float = 2.0) -> tuple[float, float]:
    candidates = np.linspace(0.08, 0.80, 145)
    best_threshold = 0.30
    best_score = -1.0
    y_array = y_true.to_numpy()
    for threshold in candidates:
        predicted = (calibrated_probs >= threshold).astype(int)
        score = fbeta_score(y_array, predicted, beta=beta, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, float(best_score)


def _select_threshold(
    y_true: pd.Series,
    calibrated_probs: np.ndarray,
    strategy: str,
    precision_floor: float | None,
    recall_floor: float | None,
    fallback_beta: float = 2.0,
) -> tuple[float, str, dict[str, float]]:
    candidates = np.linspace(0.08, 0.80, 145)
    y_array = y_true.to_numpy()

    supported_strategies = {
        "max_f1",
        "max_f1_subject_to_precision_floor",
        "precision_max_subject_to_recall_floor",
        "max_f2",
    }
    if strategy not in supported_strategies:
        raise ValueError(f"Unsupported threshold strategy '{strategy}'. Supported: {sorted(supported_strategies)}")

    best_row: dict[str, float] | None = None
    best_score = -1.0

    for threshold in candidates:
        predicted = (calibrated_probs >= threshold).astype(int)
        precision_value = float(precision_score(y_array, predicted, zero_division=0))
        recall_value = float(recall_score(y_array, predicted, zero_division=0))
        f1_value = float(f1_score(y_array, predicted, zero_division=0))
        f2_value = float(fbeta_score(y_array, predicted, beta=2.0, zero_division=0))

        if precision_floor is not None and precision_value < precision_floor:
            continue

        if recall_floor is not None and recall_value < recall_floor:
            continue

        if strategy == "precision_max_subject_to_recall_floor":
            score = precision_value
        elif strategy in {"max_f1", "max_f1_subject_to_precision_floor"}:
            score = f1_value
        else:
            score = f2_value

        if score > best_score:
            best_score = score
            best_row = {
                "threshold": float(threshold),
                "precision": precision_value,
                "recall": recall_value,
                "f1": f1_value,
                "f2": f2_value,
            }

    if best_row is not None:
        return float(best_row["threshold"]), strategy, best_row

    fallback_threshold, _ = _optimize_threshold(y_true, calibrated_probs, beta=fallback_beta)
    fallback_pred = (calibrated_probs >= fallback_threshold).astype(int)
    fallback_metrics = {
        "threshold": float(fallback_threshold),
        "precision": float(precision_score(y_array, fallback_pred, zero_division=0)),
        "recall": float(recall_score(y_array, fallback_pred, zero_division=0)),
        "f1": float(f1_score(y_array, fallback_pred, zero_division=0)),
        "f2": float(fbeta_score(y_array, fallback_pred, beta=2.0, zero_division=0)),
    }
    return fallback_threshold, "fallback_fbeta", fallback_metrics


def _derive_risk_band_thresholds(primary_threshold: float) -> dict[str, float]:
    # Medium-risk threshold is intentionally lower to support early intervention triage.
    medium_threshold = max(0.20, round(primary_threshold * 0.65, 3))
    if medium_threshold >= primary_threshold:
        medium_threshold = max(0.10, round(primary_threshold - 0.10, 3))
    return {
        "medium": float(medium_threshold),
        "high": float(primary_threshold),
    }


def main(config_path: str) -> None:
    config = load_config(config_path)
    random_state = config["model"]["random_state"]
    target_col = config["model"]["target_column"]
    target_rate = float(config["model"]["calibration_target_rate"])
    base_threshold = float(config["model"]["positive_threshold"])
    threshold_strategy = str(config["model"].get("threshold_strategy", "max_f1_subject_to_precision_floor"))
    precision_floor_cfg = config["model"].get("threshold_precision_floor")
    recall_floor_cfg = config["model"].get("threshold_recall_floor")
    precision_floor = float(precision_floor_cfg) if precision_floor_cfg is not None else None
    recall_floor = float(recall_floor_cfg) if recall_floor_cfg is not None else None

    dataset_path = _resolve_project_path(config["data"]["processed_dir"]) / "readmission_model_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Processed dataset missing: {dataset_path}")

    # Read model-ready data and split train/test with class stratification.
    df = pd.read_csv(dataset_path)
    x = df.drop(columns=[target_col])
    y = df[target_col]

    missing_required_features = [
        col for col in (NUMERIC_COLUMNS + CATEGORICAL_COLUMNS) if col not in x.columns
    ]
    if missing_required_features:
        raise KeyError(f"Dataset is missing required feature columns: {missing_required_features}")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=config["model"]["test_size"], random_state=random_state, stratify=y
    )
    x_subtrain, x_val, y_subtrain, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        random_state=random_state,
        stratify=y_train,
    )

    max_tuning_rows = int(config["model"].get("max_tuning_rows", 20000))
    if len(x_subtrain) > max_tuning_rows:
        x_subtrain, _, y_subtrain, _ = train_test_split(
            x_subtrain,
            y_subtrain,
            train_size=max_tuning_rows,
            random_state=random_state,
            stratify=y_subtrain,
        )

    print(
        f"Tuning set size: {len(x_subtrain)} rows (max_tuning_rows={max_tuning_rows}); "
        f"validation size: {len(x_val)} rows"
    )

    preprocessor = build_preprocessor()

    candidate_specs = _candidate_specifications(random_state)
    include_mlp_candidate = bool(config["model"].get("include_mlp_candidate", False))
    if not include_mlp_candidate and "deep_mlp" in candidate_specs:
        candidate_specs.pop("deep_mlp")
    include_xgboost_candidate = bool(config["model"].get("include_xgboost_candidate", HAS_XGBOOST))
    if not include_xgboost_candidate and "gradient_boosting" in candidate_specs:
        candidate_specs.pop("gradient_boosting")

    if not candidate_specs:
        raise ValueError("No candidate models enabled. Check model include_* settings in config.")

    print(f"Candidate models: {', '.join(candidate_specs.keys())}")

    candidate_rows: list[dict[str, float | str]] = []
    fitted_estimators: dict[str, Pipeline] = {}
    val_probabilities: dict[str, np.ndarray] = {}
    candidate_thresholds: dict[str, float] = {}
    candidate_threshold_metrics: dict[str, dict[str, float]] = {}
    candidate_threshold_strategies: dict[str, str] = {}
    train_prevalence = float(y_subtrain.mean())

    for model_name, spec in candidate_specs.items():
        print(f"Training candidate: {model_name}")
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", spec["estimator"]),
            ]
        )
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=spec["params"],
            n_iter=int(spec["n_iter"]),
            scoring="average_precision",
            cv=2,
            random_state=random_state,
            n_jobs=1,
            refit=True,
        )
        search.fit(x_subtrain, y_subtrain)

        best_pipe: Pipeline = search.best_estimator_
        val_raw_probs = best_pipe.predict_proba(x_val)[:, 1]
        val_calibrated_probs = prevalence_shift_calibration(
            probabilities=np.array(val_raw_probs),
            train_prevalence=train_prevalence,
            target_prevalence=target_rate,
        )
        val_metrics = _classification_metrics(y_val, val_calibrated_probs, threshold=base_threshold)

        candidate_threshold, candidate_strategy_used, candidate_threshold_metric = _select_threshold(
            y_val,
            val_calibrated_probs,
            strategy=threshold_strategy,
            precision_floor=precision_floor,
            recall_floor=recall_floor,
            fallback_beta=2.0,
        )

        fitted_estimators[model_name] = best_pipe
        val_probabilities[model_name] = val_calibrated_probs
        candidate_thresholds[model_name] = candidate_threshold
        candidate_threshold_metrics[model_name] = candidate_threshold_metric
        candidate_threshold_strategies[model_name] = candidate_strategy_used
        candidate_rows.append(
            {
                "model": model_name,
                "cv_best_pr_auc": float(search.best_score_),
                "val_roc_auc": val_metrics["roc_auc"],
                "val_pr_auc": val_metrics["pr_auc"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_opt_threshold": float(candidate_threshold),
                "val_precision_at_opt_threshold": float(candidate_threshold_metric["precision"]),
                "val_recall_at_opt_threshold": float(candidate_threshold_metric["recall"]),
                "val_f1_at_opt_threshold": float(candidate_threshold_metric["f1"]),
                "val_f2_at_opt_threshold": float(candidate_threshold_metric["f2"]),
            }
        )

    metrics_df = pd.DataFrame(candidate_rows)
    selection_metric = str(config["model"].get("selection_metric", "val_f1_at_opt_threshold"))
    if selection_metric not in metrics_df.columns:
        selection_metric = "val_f1_at_opt_threshold"

    metrics_df = metrics_df.sort_values(by=[selection_metric, "val_pr_auc", "val_roc_auc"], ascending=False)
    best_name = str(metrics_df.iloc[0]["model"])
    best_model = fitted_estimators[best_name]

    tuned_threshold = float(candidate_thresholds[best_name])
    realized_threshold_metrics = candidate_threshold_metrics[best_name]
    threshold_strategy_used = candidate_threshold_strategies[best_name]
    tuned_f2 = float(realized_threshold_metrics["f2"])
    realized_recall = float(realized_threshold_metrics["recall"])
    risk_band_thresholds = _derive_risk_band_thresholds(tuned_threshold)

    # Refit winning architecture on full training set after hyperparameter search.
    best_model.fit(x_train, y_train)

    # Calibrate and evaluate on held-out test set.
    raw_probs_test = best_model.predict_proba(x_test)[:, 1]
    calibrated_probs_test = prevalence_shift_calibration(
        probabilities=np.array(raw_probs_test),
        train_prevalence=float(y_train.mean()),
        target_prevalence=target_rate,
    )
    test_metrics = _classification_metrics(y_test, calibrated_probs_test, threshold=tuned_threshold)
    test_metrics_base_threshold = _classification_metrics(y_test, calibrated_probs_test, threshold=base_threshold)

    artifacts_dir = _resolve_project_path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = _resolve_project_path(config["app"]["model_path"])
    metadata_path = _resolve_project_path(config["app"]["metadata_path"])
    metrics_path = artifacts_dir / "model_comparison.csv"

    # Save all outputs needed by dashboard and future reproducibility.
    joblib.dump(best_model, model_path)
    metrics_df.to_csv(metrics_path, index=False)

    metadata = {
        "best_model": str(best_name),
        "train_prevalence": float(y_train.mean()),
        "target_prevalence": target_rate,
        "threshold": float(tuned_threshold),
        "threshold_strategy": threshold_strategy_used,
        "requested_threshold_strategy": threshold_strategy,
        "threshold_precision_floor": precision_floor,
        "threshold_recall_floor": recall_floor,
        "validation_precision_at_threshold": float(realized_threshold_metrics["precision"]),
        "validation_recall_at_threshold": float(realized_recall),
        "validation_f1_at_threshold": float(realized_threshold_metrics["f1"]),
        "validation_f2_at_threshold": float(tuned_f2),
        "risk_band_thresholds": risk_band_thresholds,
        "test_metrics": test_metrics,
        "test_metrics_base_threshold": test_metrics_base_threshold,
        "features": list(x.columns),
        "numeric_features": NUMERIC_COLUMNS,
        "categorical_features": CATEGORICAL_COLUMNS,
        "selection_metric": selection_metric,
        "xgboost_available": HAS_XGBOOST,
    }

    with metadata_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(metadata, file, sort_keys=False)

    print("Training complete.")
    print(f"Best model: {best_name}")
    print(f"Optimized threshold: {tuned_threshold:.3f} (validation F2={tuned_f2:.4f})")
    print(
        "Validation threshold metrics: "
        f"precision={realized_threshold_metrics['precision']:.4f}, "
        f"recall={realized_threshold_metrics['recall']:.4f}, "
        f"f1={realized_threshold_metrics['f1']:.4f}"
    )
    print("Held-out test metrics:")
    for metric_name, metric_value in test_metrics.items():
        print(f"  - {metric_name}: {metric_value:.4f}")
    print(f"Saved model to: {model_path}")
    print(f"Saved metadata to: {metadata_path}")
    print(f"Saved comparison metrics to: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train readmission models and persist best artifact.")
    parser.add_argument("--config", default="config/project_config.yaml", help="Path to project config")
    args = parser.parse_args()
    main(args.config)

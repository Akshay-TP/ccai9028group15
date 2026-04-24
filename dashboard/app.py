from __future__ import annotations

import sys
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if __package__ in (None, ""):
    # Support direct file execution from dashboard folder in VS Code.
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.api.registry import delete_patient, initialize_registry, list_patients, upsert_patient
from src.models.calibration import prevalence_shift_calibration
from src.models.inference import ReadmissionScorer
from src.utils import load_config

from src.api.registry import list_audit_events

ADMISSION_TYPE_LABEL_TO_CODE = {
    "Emergency": "1",
    "Urgent": "2",
    "Elective": "3",
    "Unknown / Not Recorded": "6",
    "Other / Unspecified": "8",
}

DISCHARGE_LABEL_TO_CODE = {
    "Discharged to home": "1",
    "Transferred to short term hospital": "2",
    "Transferred to SNF": "3",
    "Home with home health service": "6",
    "Left against medical advice": "7",
    "Expired": "11",
    "Unknown / Not Recorded": "18",
}

ADMISSION_SOURCE_LABEL_TO_CODE = {
    "Physician referral": "1",
    "Clinic referral": "2",
    "HMO referral": "3",
    "Transfer from hospital": "4",
    "Emergency room": "7",
    "Unknown / Not available": "9",
}


def select_code(label: str, label_to_code: dict[str, str], default_code: str) -> str:
    labels = list(label_to_code.keys())
    default_index = 0
    for i, text_label in enumerate(labels):
        if label_to_code[text_label] == default_code:
            default_index = i
            break
    chosen_label = st.selectbox(label, labels, index=default_index)
    return label_to_code[chosen_label]


def decode_code_column(series: pd.Series, label_to_code: dict[str, str]) -> pd.Series:
    code_to_label = {code: label for label, code in label_to_code.items()}
    return series.astype(str).map(code_to_label).fillna("Other / Unlisted")


def train_portable_fallback_model(metadata_path: str) -> tuple[Pipeline, dict]:
    """Train a lightweight fallback model when xgboost is unavailable in the current environment."""
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "processed" / "readmission_model_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Fallback dataset not found at {dataset_path}. Run data prep and training pipeline first."
        )

    with Path(metadata_path).open("r", encoding="utf-8") as file:
        metadata = yaml.safe_load(file)

    df = pd.read_csv(dataset_path)
    target_col = "readmitted_30d"
    features = metadata.get("features", [c for c in df.columns if c != target_col])
    numeric_features = metadata.get("numeric_features", [])
    categorical_features = metadata.get("categorical_features", [])

    x = df[features]
    y = df[target_col]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    portable_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    portable_model.fit(x, y)
    return portable_model, metadata


def score_with_portable_model(model: Pipeline, metadata: dict, patient_rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(patient_rows)
    probs = model.predict_proba(df)[:, 1]
    calibrated = prevalence_shift_calibration(
        probabilities=np.array(probs),
        train_prevalence=float(metadata.get("train_prevalence", 0.12)),
        target_prevalence=float(metadata.get("target_prevalence", 0.17)),
    )
    threshold = float(metadata.get("threshold", 0.30))
    medium_threshold = float(metadata.get("risk_band_thresholds", {}).get("medium", max(0.10, threshold * 0.65)))
    labels = np.where(calibrated >= threshold, "HIGH", "LOW")
    bands = np.where(calibrated >= threshold, "HIGH", np.where(calibrated >= medium_threshold, "MEDIUM", "LOW"))

    output = df.copy()
    output["raw_probability"] = probs
    output["calibrated_probability"] = calibrated
    output["risk_label"] = labels
    output["risk_band"] = bands
    return output


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def enrich_registry_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    if "total_prior_visits" not in enriched.columns:
        enriched["total_prior_visits"] = (
            pd.to_numeric(enriched.get("number_outpatient", 0), errors="coerce").fillna(0)
            + pd.to_numeric(enriched.get("number_emergency", 0), errors="coerce").fillna(0)
            + pd.to_numeric(enriched.get("number_inpatient", 0), errors="coerce").fillna(0)
        )
    if "inpatient_visit_ratio" not in enriched.columns:
        enriched["inpatient_visit_ratio"] = (
            pd.to_numeric(enriched.get("number_inpatient", 0), errors="coerce").fillna(0)
            / (pd.to_numeric(enriched.get("total_prior_visits", 0), errors="coerce").fillna(0) + 1.0)
        )
    if "medications_per_day" not in enriched.columns:
        enriched["medications_per_day"] = (
            pd.to_numeric(enriched.get("num_medications", 0), errors="coerce").fillna(0)
            / (pd.to_numeric(enriched.get("time_in_hospital", 0), errors="coerce").fillna(0) + 1.0)
        )
    if "age_over_65" not in enriched.columns:
        enriched["age_over_65"] = (
            pd.to_numeric(enriched.get("age_midpoint", 0), errors="coerce").fillna(0) >= 65
        ).astype(int)
    return enriched


def build_follow_up_recommendation(row: pd.Series) -> str:
    suggestions: list[str] = []
    risk_band = str(row.get("risk_band", row.get("risk_label", "LOW")))
    if risk_band == "HIGH":
        suggestions.append("Call patient within 48h")
        suggestions.append("Schedule nurse tele-check within 7d")
    elif risk_band == "MEDIUM":
        suggestions.append("Call patient within 5d")
        suggestions.append("Medication adherence check within 14d")
    if float(row.get("number_inpatient", 0)) >= 2:
        suggestions.append("Coordinate post-discharge care plan")
    if int(row.get("flag_heart_failure", 0)) == 1 or int(row.get("flag_kidney_disease", 0)) == 1:
        suggestions.append("Escalate to chronic-care clinic")
    if not suggestions:
        suggestions.append("Standard follow-up workflow")
    return " | ".join(suggestions)


@st.cache_resource
def load_primary_scorer(model_path: str, metadata_path: str) -> ReadmissionScorer:
    return ReadmissionScorer(model_path=model_path, metadata_path=metadata_path)


def score_patients(feature_df: pd.DataFrame, model_path: str, metadata_path: str) -> tuple[pd.DataFrame, str]:
    enriched = enrich_registry_features(feature_df)
    try:
        scorer = load_primary_scorer(model_path=model_path, metadata_path=metadata_path)
        scored = scorer.score(enriched.to_dict(orient="records"))
        return scored, "trained_model"
    except ModuleNotFoundError as exc:
        missing_module = str(exc).split("'")[1] if "'" in str(exc) else "required dependency"
        st.warning(
            f"Missing dependency: {missing_module}. "
            "Using portable logistic fallback model for this session."
        )
        portable_model, portable_metadata = train_portable_fallback_model(metadata_path)
        scored = score_with_portable_model(
            model=portable_model,
            metadata=portable_metadata,
            patient_rows=enriched.to_dict(orient="records"),
        )
        return scored, "portable_fallback"


@st.cache_resource
def load_model_artifact(model_path: str) -> Pipeline:
    import joblib

    return joblib.load(model_path)


@st.cache_data
def build_feature_importance_table(model_path: str) -> pd.DataFrame:
    try:
        model = load_model_artifact(model_path)
        preprocessor = model.named_steps.get("preprocessor")
        estimator = model.named_steps.get("model")
        if preprocessor is None or estimator is None:
            return pd.DataFrame()

        feature_names = preprocessor.get_feature_names_out()
        if hasattr(estimator, "coef_"):
            raw = np.abs(np.ravel(estimator.coef_))
        elif hasattr(estimator, "feature_importances_"):
            raw = np.asarray(estimator.feature_importances_)
        else:
            return pd.DataFrame()

        importance = pd.DataFrame(
            {
                "feature": [str(name).replace("num__", "").replace("cat__", "") for name in feature_names],
                "importance": raw,
            }
        ).sort_values("importance", ascending=False)

        total = float(importance["importance"].sum())
        if total > 0:
            importance["importance_share"] = importance["importance"] / total
        else:
            importance["importance_share"] = 0.0
        return importance
    except Exception:
        return pd.DataFrame()


# Basic page metadata.
st.set_page_config(page_title="Hospital Readmission Predictor - Diabetes", page_icon="🏥", layout="wide")

# Custom CSS for a coherent high-contrast clinical dashboard look.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=Source+Serif+4:opsz,wght@8..60,600&display=swap');

    :root {
        --bg-1: #eef4f8;
        --bg-2: #dfeaf2;
        --accent: #005a9c;
        --accent-2: #9a5d00;
        --alert: #b00020;
        --card: #ffffff;
        --text: #0a1220;
        --muted: #2f455b;
    }

    .stApp {
        font-family: 'Manrope', sans-serif;
        background: radial-gradient(circle at 15% -5%, #c5dff0, var(--bg-1) 58%);
        color: var(--text);
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3, h4 {
        font-family: 'Source Serif 4', serif;
    }

    h1, h2, h3, h4, h5, h6, p, label, span, div, li {
        color: var(--text) !important;
    }

    .stCaptionContainer p {
        color: var(--muted) !important;
    }

    .metric-tile {
        border-radius: 16px;
        border: 1px solid rgba(0, 90, 156, 0.45);
        background: var(--card);
        padding: 16px;
        box-shadow: 0 8px 20px rgba(10, 18, 32, 0.14);
    }

    .title-banner {
        background: linear-gradient(110deg, var(--accent), var(--accent-2));
        color: #ffffff;
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 14px;
        animation: fadeIn 0.8s ease-out;
        box-shadow: 0 14px 30px rgba(10, 18, 32, 0.28);
    }

    .title-banner h2, .title-banner p {
        color: #ffffff !important;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

config = load_config()
db_path = config["app"]["database_path"]
model_path = config["app"]["model_path"]
metadata_path = config["app"]["metadata_path"]

# Ensure local registry exists before any UI operation.
initialize_registry(db_path)

st.markdown(
    """
    <div class="title-banner">
        <h2>Hospital Readmission Predictor - Diabetes</h2>
        <p>Prototype dashboard for 30-day unplanned diabetes-related readmission triage.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(f"Updated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

patients_df = list_patients(db_path)

with st.sidebar:
    st.header("Filter Cohort")
    search_text = st.text_input("Patient ID contains", value="")
    genders = sorted(patients_df["gender"].dropna().astype(str).unique().tolist()) if not patients_df.empty else []
    selected_genders = st.multiselect("Gender", options=genders, default=genders)

    if patients_df.empty:
        age_min, age_max = (20, 95)
    else:
        age_min = int(max(20, np.floor(patients_df["age_midpoint"].min())))
        age_max = int(min(95, np.ceil(patients_df["age_midpoint"].max())))
    selected_age = st.slider("Age range", min_value=20, max_value=95, value=(age_min, age_max))
    comorbidity_only = st.checkbox("Only with >=1 comorbidity", value=False)

filtered_patients = patients_df.copy()
if not filtered_patients.empty:
    if search_text.strip():
        filtered_patients = filtered_patients[
            filtered_patients["patient_id"].str.contains(search_text.strip(), case=False, na=False)
        ]
    if selected_genders:
        filtered_patients = filtered_patients[filtered_patients["gender"].isin(selected_genders)]
    filtered_patients = filtered_patients[
        filtered_patients["age_midpoint"].between(selected_age[0], selected_age[1], inclusive="both")
    ]
    if comorbidity_only:
        filtered_patients = filtered_patients[
            (filtered_patients["flag_diabetes"].fillna(0).astype(int)
             + filtered_patients["flag_heart_failure"].fillna(0).astype(int)
             + filtered_patients["flag_kidney_disease"].fillna(0).astype(int)
             + filtered_patients["flag_copd"].fillna(0).astype(int))
            >= 1
        ]

metric_a, metric_b, metric_c, metric_d = st.columns(4)
metric_a.markdown(
    f"<div class='metric-tile'><h4>Total Patients</h4><h2>{len(patients_df)}</h2></div>",
    unsafe_allow_html=True,
)
metric_b.markdown(
    f"<div class='metric-tile'><h4>Filtered Cohort</h4><h2>{len(filtered_patients)}</h2></div>",
    unsafe_allow_html=True,
)
avg_age = float(patients_df["age_midpoint"].mean()) if not patients_df.empty else 0.0
metric_c.markdown(
    f"<div class='metric-tile'><h4>Average Age</h4><h2>{avg_age:.1f}</h2></div>",
    unsafe_allow_html=True,
)
high_acuity = int((patients_df["number_inpatient"].fillna(0) >= 2).sum()) if not patients_df.empty else 0
metric_d.markdown(
    f"<div class='metric-tile'><h4>Frequent Inpatients</h4><h2>{high_acuity}</h2></div>",
    unsafe_allow_html=True,
)

tab_registry, tab_scoring, tab_audit, tab_model = st.tabs(
    ["Patient Registry", "Risk Scoring", "Audit Trail", "Model Card"]
)

with tab_registry:
    left, right = st.columns([1.3, 1.0])

    with left:
        st.subheader("Add or Update Patient")
        with st.form("patient_form", clear_on_submit=False):
            patient_id = st.text_input("Patient ID", value="HK-0100")
            race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            age_midpoint = st.slider("Age", min_value=20, max_value=95, value=65)

            admission_type_id = select_code("Admission Type", ADMISSION_TYPE_LABEL_TO_CODE, "1")
            discharge_disposition_id = select_code("Discharge Disposition", DISCHARGE_LABEL_TO_CODE, "1")
            admission_source_id = select_code("Admission Source", ADMISSION_SOURCE_LABEL_TO_CODE, "7")

            time_in_hospital = st.slider("Length of Stay (days)", 1, 14, 4)
            num_lab_procedures = st.slider("Lab Procedures", 1, 130, 45)
            num_procedures = st.slider("Procedures", 0, 8, 1)
            num_medications = st.slider("Medications", 1, 60, 12)
            number_outpatient = st.slider("Outpatient Visits", 0, 20, 1)
            number_emergency = st.slider("Emergency Visits", 0, 10, 0)
            number_inpatient = st.slider("Prior Inpatient Visits", 0, 12, 1)
            number_diagnoses = st.slider("Diagnoses Count", 1, 16, 8)

            A1Cresult = st.selectbox("A1C Result", ["None", ">7", ">8", "Norm"])
            max_glu_serum = st.selectbox("Max Glucose", ["None", ">200", ">300", "Norm"])
            insulin = st.selectbox("Insulin Change", ["No", "Steady", "Up", "Down"])
            change = st.selectbox("Medication Change", ["No", "Ch"])
            diabetesMed = st.selectbox("On Diabetes Med", ["No", "Yes"])

            flag_diabetes = st.checkbox("Diabetes Comorbidity", value=True)
            flag_heart_failure = st.checkbox("Heart Failure Comorbidity", value=False)
            flag_kidney_disease = st.checkbox("Kidney Disease Comorbidity", value=False)
            flag_copd = st.checkbox("COPD Comorbidity", value=False)

            submitted = st.form_submit_button("Save Patient", type="primary")

        if submitted:
            upsert_patient(
                db_path,
                {
                    "patient_id": patient_id,
                    "race": race,
                    "gender": gender,
                    "age_midpoint": age_midpoint,
                    "admission_type_id": admission_type_id,
                    "discharge_disposition_id": discharge_disposition_id,
                    "admission_source_id": admission_source_id,
                    "time_in_hospital": time_in_hospital,
                    "num_lab_procedures": num_lab_procedures,
                    "num_procedures": num_procedures,
                    "num_medications": num_medications,
                    "number_outpatient": number_outpatient,
                    "number_emergency": number_emergency,
                    "number_inpatient": number_inpatient,
                    "number_diagnoses": number_diagnoses,
                    "A1Cresult": A1Cresult,
                    "max_glu_serum": max_glu_serum,
                    "insulin": insulin,
                    "change": change,
                    "diabetesMed": diabetesMed,
                    "flag_diabetes": int(flag_diabetes),
                    "flag_heart_failure": int(flag_heart_failure),
                    "flag_kidney_disease": int(flag_kidney_disease),
                    "flag_copd": int(flag_copd),
                },
                actor="dashboard_admin",
            )
            st.success(f"Saved {patient_id}.")
            st.rerun()

    with right:
        st.subheader("Delete Patient")
        if patients_df.empty:
            st.info("No patients in the registry.")
        else:
            remove_id = st.selectbox("Select Patient ID", patients_df["patient_id"].tolist())
            if st.button("Delete Selected", type="secondary"):
                delete_patient(db_path, remove_id, actor="dashboard_admin")
                st.warning(f"Deleted {remove_id}.")
                st.rerun()

    st.subheader("Registry Table")
    display_df = filtered_patients.copy()
    if not display_df.empty:
        display_df["admission_type_text"] = decode_code_column(
            display_df["admission_type_id"], ADMISSION_TYPE_LABEL_TO_CODE
        )
        display_df["discharge_text"] = decode_code_column(
            display_df["discharge_disposition_id"], DISCHARGE_LABEL_TO_CODE
        )
        display_df["admission_source_text"] = decode_code_column(
            display_df["admission_source_id"], ADMISSION_SOURCE_LABEL_TO_CODE
        )
    st.dataframe(display_df, width="stretch", height=300)

    st.download_button(
        "Download Filtered Registry CSV",
        data=to_csv_bytes(display_df),
        file_name="filtered_registry.csv",
        mime="text/csv",
        disabled=display_df.empty,
    )

with tab_scoring:
    st.subheader("Run Model Scoring")
    if patients_df.empty:
        st.info("No patients available. Add records in the Patient Registry tab first.")
    elif not (pd.io.common.file_exists(model_path) and pd.io.common.file_exists(metadata_path)):
        st.error("Model artifact missing. Run training pipeline first.")
    else:
        scoring_scope = st.radio(
            "Scoring Scope",
            options=["Filtered Cohort", "All Patients"],
            horizontal=True,
            index=0,
        )
        scope_df = filtered_patients if scoring_scope == "Filtered Cohort" else patients_df

        if st.button("Score Selected Patients", type="primary", disabled=scope_df.empty):
            feature_df = scope_df.drop(columns=[col for col in ["patient_id", "created_at", "updated_at"] if col in scope_df])
            scored, scoring_mode = score_patients(feature_df, model_path=model_path, metadata_path=metadata_path)
            scored.insert(0, "patient_id", scope_df["patient_id"].tolist())
            scored["recommended_action"] = scored.apply(build_follow_up_recommendation, axis=1)
            st.session_state["scored_df"] = scored
            st.session_state["scoring_mode"] = scoring_mode
            st.session_state["scoring_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        scored_df: pd.DataFrame | None = st.session_state.get("scored_df")
        if scored_df is not None and not scored_df.empty:
            st.caption(
                f"Last scoring run: {st.session_state.get('scoring_time', 'n/a')} "
                f"({st.session_state.get('scoring_mode', 'trained_model')})"
            )
            k1, k2, k3 = st.columns(3)
            if "risk_band" not in scored_df.columns:
                scored_df["risk_band"] = scored_df["risk_label"]
            high_count = int((scored_df["risk_band"] == "HIGH").sum())
            medium_count = int((scored_df["risk_band"] == "MEDIUM").sum())
            high_share = (high_count / len(scored_df)) * 100
            medium_share = (medium_count / len(scored_df)) * 100
            avg_risk = float(scored_df["calibrated_probability"].mean())
            k1.metric("High-Risk Patients", f"{high_count}")
            k2.metric("High + Medium Share", f"{(high_share + medium_share):.1f}%")
            k3.metric("Average Calibrated Risk", f"{avg_risk:.3f}")

            st.dataframe(
                scored_df[
                    [
                        "patient_id",
                        "calibrated_probability",
                        "risk_band",
                        "risk_label",
                        "recommended_action",
                    ]
                ],
                width="stretch",
                height=260,
            )

            chart_a, chart_b = st.columns(2)
            with chart_a:
                hist = px.histogram(
                    scored_df,
                    x="calibrated_probability",
                    color="risk_band",
                    nbins=20,
                    title="Calibrated 30-Day Risk Distribution",
                    color_discrete_map={"HIGH": "#b00020", "MEDIUM": "#9a5d00", "LOW": "#005a9c"},
                )
                st.plotly_chart(hist, width="stretch")

            with chart_b:
                risk_mix = (
                    scored_df.groupby("risk_band", as_index=False)["patient_id"].count().rename(columns={"patient_id": "count"})
                )
                bar = px.bar(
                    risk_mix,
                    x="risk_band",
                    y="count",
                    color="risk_band",
                    title="Risk Band Mix",
                    color_discrete_map={"HIGH": "#b00020", "MEDIUM": "#9a5d00", "LOW": "#005a9c"},
                )
                st.plotly_chart(bar, width="stretch")

            download_name = f"scored_patients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            st.download_button(
                "Download Scored Cohort CSV",
                data=to_csv_bytes(scored_df),
                file_name=download_name,
                mime="text/csv",
            )
        else:
            st.info("Run scoring to generate risk analytics and intervention recommendations.")

with tab_audit:
    st.subheader("Patient Registry Audit Events")
    audit_df = list_audit_events(db_path, limit=500)
    if audit_df.empty:
        st.info("No audit events yet.")
    else:
        st.dataframe(audit_df, width="stretch", height=320)
        st.download_button(
            "Download Audit Log CSV",
            data=to_csv_bytes(audit_df),
            file_name="patient_audit_log.csv",
            mime="text/csv",
        )

with tab_model:
    st.subheader("Model Metadata and Comparison")
    metadata: dict[str, Any] = {}
    if pd.io.common.file_exists(metadata_path):
        with Path(metadata_path).open("r", encoding="utf-8") as file:
            metadata = yaml.safe_load(file) or {}

    comparison_path = Path(model_path).resolve().parent / "model_comparison.csv"
    if comparison_path.exists():
        comparison_df = pd.read_csv(comparison_path)
        st.dataframe(comparison_df, width="stretch")
    else:
        st.info("Model comparison file not found yet.")

    importance_df = build_feature_importance_table(model_path)
    if not importance_df.empty:
        st.markdown("### Top Feature Drivers")
        top_importance = importance_df.head(15).copy()
        chart = px.bar(
            top_importance.sort_values("importance_share", ascending=True),
            x="importance_share",
            y="feature",
            orientation="h",
            title="Model Feature Importance Share",
            color_discrete_sequence=["#005a9c"],
        )
        st.plotly_chart(chart, width="stretch")
        st.dataframe(top_importance, width="stretch")
    else:
        st.info("Feature importance unavailable for current model type.")

    if metadata:
        st.json(metadata)
    else:
        st.info("Model metadata file not found yet.")

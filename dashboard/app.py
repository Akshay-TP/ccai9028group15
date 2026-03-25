from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

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
    labels = np.where(calibrated >= threshold, "HIGH", "LOW")

    output = df.copy()
    output["raw_probability"] = probs
    output["calibrated_probability"] = calibrated
    output["risk_label"] = labels
    return output
# Basic page metadata.
st.set_page_config(page_title="HA Readmission Risk Dashboard", page_icon="🏥", layout="wide")

# Custom CSS for a more polished and readable student project dashboard.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --bg-1: #070b12;
        --bg-2: #0a1422;
        --accent: #00d1ff;
        --accent-2: #20ffa8;
        --alert: #ff5d73;
        --card: #0f1a2b;
        --text: #f5fbff;
        --muted: #9fb3c8;
    }

    .stApp {
        font-family: 'Space Grotesk', sans-serif;
        background: radial-gradient(circle at 12% 8%, #0e2237, var(--bg-1) 58%);
        color: var(--text);
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3, h4, h5, h6, p, label, span, div {
        color: var(--text) !important;
    }

    .stCaptionContainer p {
        color: var(--muted) !important;
    }

    .metric-tile {
        border-radius: 16px;
        border: 1px solid rgba(0, 209, 255, 0.28);
        background: var(--card);
        padding: 16px;
        box-shadow: 0 10px 26px rgba(0, 0, 0, 0.45);
    }

    .title-banner {
        background: linear-gradient(120deg, var(--accent), var(--accent-2));
        color: #02131f;
        border-radius: 18px;
        padding: 20px;
        margin-bottom: 14px;
        animation: fadeIn 0.8s ease-out;
        box-shadow: 0 14px 34px rgba(0, 209, 255, 0.24);
    }

    .title-banner h2, .title-banner p {
        color: #02131f !important;
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
        <h2>Hong Kong Public Hospital Readmission Intelligence</h2>
        <p>Prototype dashboard for 30-day unplanned readmission triage in chronic-disease cohorts.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(f"Updated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

left, right = st.columns([1, 1])

# Left side: create or update patient records.
with left:
    st.subheader("Admin Panel: Add or Update Patient")
    with st.form("patient_form", clear_on_submit=False):
        patient_id = st.text_input("Patient ID", value="P-0001")
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

        submitted = st.form_submit_button("Save Patient")

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
        )
        st.success(f"Saved {patient_id}.")

with right:
    # Right side: delete flow to keep admin actions explicit.
    st.subheader("Admin Panel: Delete Patient")
    patients_df = list_patients(db_path)
    if patients_df.empty:
        st.info("No patients yet. Add one on the left.")
    else:
        remove_id = st.selectbox("Select Patient ID", patients_df["patient_id"].tolist())
        if st.button("Delete Selected", type="secondary"):
            delete_patient(db_path, remove_id)
            st.warning(f"Deleted {remove_id}.")

st.subheader("Registered Patients")
patients_df = list_patients(db_path)
display_df = patients_df.copy()
if not display_df.empty:
    display_df["admission_type_text"] = decode_code_column(display_df["admission_type_id"], ADMISSION_TYPE_LABEL_TO_CODE)
    display_df["discharge_text"] = decode_code_column(display_df["discharge_disposition_id"], DISCHARGE_LABEL_TO_CODE)
    display_df["admission_source_text"] = decode_code_column(display_df["admission_source_id"], ADMISSION_SOURCE_LABEL_TO_CODE)
st.dataframe(display_df, use_container_width=True, height=250)

if patients_df.empty:
    st.stop()

st.subheader("Run Model Scoring")
if not (pd.io.common.file_exists(model_path) and pd.io.common.file_exists(metadata_path)):
    st.error("Model artifact missing. Run training pipeline first.")
    st.stop()

if st.button("Score All Patients", type="primary"):
    try:
        # Load scoring engine only when needed so the page still renders if an optional dependency is missing.
        scorer = ReadmissionScorer(model_path=model_path, metadata_path=metadata_path)
        use_portable_fallback = False
        portable_model = None
        portable_metadata = None
    except ModuleNotFoundError as exc:
        missing_module = str(exc).split("'")[1] if "'" in str(exc) else "required dependency"
        st.warning(
            f"Missing dependency: {missing_module}. "
            "Using portable logistic fallback model for this session."
        )
        portable_model, portable_metadata = train_portable_fallback_model(metadata_path)
        use_portable_fallback = True

    # patient_id is not a model feature, so remove before scoring.
    feature_df = patients_df.drop(columns=["patient_id"])
    if use_portable_fallback:
        scored = score_with_portable_model(
            model=portable_model,
            metadata=portable_metadata,
            patient_rows=feature_df.to_dict(orient="records"),
        )
    else:
        scored = scorer.score(feature_df.to_dict(orient="records"))
    scored.insert(0, "patient_id", patients_df["patient_id"].tolist())

    st.success("Scoring complete.")
    st.dataframe(scored[["patient_id", "calibrated_probability", "risk_label"]], use_container_width=True)

    fig = px.histogram(
        scored,
        x="calibrated_probability",
        color="risk_label",
        nbins=20,
        title="Distribution of Calibrated 30-Day Readmission Risk",
        color_discrete_map={"HIGH": "#e76f51", "LOW": "#2a9d8f"},
    )
    st.plotly_chart(fig, use_container_width=True)

    high_risk = scored[scored["risk_label"] == "HIGH"]
    st.markdown("### Suggested Follow-up Actions")
    if high_risk.empty:
        st.info("No high-risk patients under the current threshold.")
    else:
        st.write(f"{len(high_risk)} high-risk patients identified.")
        st.write("Recommended: trigger follow-up call in 48h, schedule nurse home visit, offer telehealth check in 7 days.")

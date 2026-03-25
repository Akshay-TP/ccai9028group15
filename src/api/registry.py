from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

"""SQLite registry used by the dashboard for patient CRUD operations."""

PATIENT_COLUMNS = [
    "patient_id",
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
]


def get_connection(db_path: str) -> sqlite3.Connection:
    # Ensure folder exists so sqlite can create the DB file.
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def initialize_registry(db_path: str) -> None:
    # Create table once; no-op if already created.
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                race TEXT,
                gender TEXT,
                age_midpoint REAL,
                admission_type_id TEXT,
                discharge_disposition_id TEXT,
                admission_source_id TEXT,
                time_in_hospital REAL,
                num_lab_procedures REAL,
                num_procedures REAL,
                num_medications REAL,
                number_outpatient REAL,
                number_emergency REAL,
                number_inpatient REAL,
                number_diagnoses REAL,
                A1Cresult TEXT,
                max_glu_serum TEXT,
                insulin TEXT,
                change TEXT,
                diabetesMed TEXT,
                flag_diabetes INTEGER,
                flag_heart_failure INTEGER,
                flag_kidney_disease INTEGER,
                flag_copd INTEGER
            )
            """
        )
        conn.commit()


def upsert_patient(db_path: str, patient: dict[str, Any]) -> None:
    initialize_registry(db_path)
    with get_connection(db_path) as conn:
        # Upsert lets us use one action for create or update by patient_id.
        placeholders = ", ".join(["?"] * len(PATIENT_COLUMNS))
        column_sql = ", ".join(PATIENT_COLUMNS)
        update_sql = ", ".join([f"{col}=excluded.{col}" for col in PATIENT_COLUMNS if col != "patient_id"])
        conn.execute(
            f"""
            INSERT INTO patients ({column_sql})
            VALUES ({placeholders})
            ON CONFLICT(patient_id) DO UPDATE SET
            {update_sql}
            """,
            [patient.get(col) for col in PATIENT_COLUMNS],
        )
        conn.commit()


def delete_patient(db_path: str, patient_id: str) -> None:
    # Remove one record by primary key.
    initialize_registry(db_path)
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM patients WHERE patient_id = ?", (patient_id,))
        conn.commit()


def list_patients(db_path: str) -> pd.DataFrame:
    # Return stable ordering for easy dashboard viewing.
    initialize_registry(db_path)
    with get_connection(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM patients ORDER BY patient_id", conn)
    return df

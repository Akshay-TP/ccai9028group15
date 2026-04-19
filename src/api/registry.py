from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
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
                flag_copd INTEGER,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patient_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_ts TEXT NOT NULL,
                action TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                actor TEXT NOT NULL,
                details TEXT
            )
            """
        )

        existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(patients)").fetchall()}
        if "created_at" not in existing_columns:
            conn.execute("ALTER TABLE patients ADD COLUMN created_at TEXT")
            conn.execute("UPDATE patients SET created_at = datetime('now') WHERE created_at IS NULL")
        if "updated_at" not in existing_columns:
            conn.execute("ALTER TABLE patients ADD COLUMN updated_at TEXT")
            conn.execute("UPDATE patients SET updated_at = datetime('now') WHERE updated_at IS NULL")

        conn.commit()


def _log_event(conn: sqlite3.Connection, action: str, patient_id: str, actor: str, details: str | None = None) -> None:
    conn.execute(
        """
        INSERT INTO patient_audit_log (event_ts, action, patient_id, actor, details)
        VALUES (?, ?, ?, ?, ?)
        """,
        (datetime.now(timezone.utc).isoformat(), action, patient_id, actor, details),
    )


def upsert_patient(db_path: str, patient: dict[str, Any], actor: str = "dashboard_user") -> None:
    initialize_registry(db_path)
    patient_id = str(patient.get("patient_id", "")).strip()
    if not patient_id:
        raise ValueError("patient_id is required.")

    with get_connection(db_path) as conn:
        exists = conn.execute(
            "SELECT 1 FROM patients WHERE patient_id = ? LIMIT 1",
            (patient_id,),
        ).fetchone() is not None

        # Upsert lets us use one action for create or update by patient_id.
        placeholders = ", ".join(["?"] * len(PATIENT_COLUMNS))
        column_sql = ", ".join(PATIENT_COLUMNS)
        update_sql = ", ".join([f"{col}=excluded.{col}" for col in PATIENT_COLUMNS if col != "patient_id"])
        now_utc = datetime.now(timezone.utc).isoformat()
        conn.execute(
            f"""
            INSERT INTO patients ({column_sql}, created_at, updated_at)
            VALUES ({placeholders}, ?, ?)
            ON CONFLICT(patient_id) DO UPDATE SET
            {update_sql},
            updated_at=excluded.updated_at
            """,
            [patient.get(col) for col in PATIENT_COLUMNS] + [now_utc, now_utc],
        )
        action = "UPDATE" if exists else "CREATE"
        _log_event(conn, action=action, patient_id=patient_id, actor=actor)
        conn.commit()


def delete_patient(db_path: str, patient_id: str, actor: str = "dashboard_user") -> None:
    # Remove one record by primary key.
    initialize_registry(db_path)
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM patients WHERE patient_id = ?", (patient_id,))
        _log_event(conn, action="DELETE", patient_id=patient_id, actor=actor)
        conn.commit()


def list_patients(db_path: str) -> pd.DataFrame:
    # Return stable ordering for easy dashboard viewing.
    initialize_registry(db_path)
    with get_connection(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM patients ORDER BY updated_at DESC, patient_id", conn)
    return df


def list_audit_events(db_path: str, limit: int = 200) -> pd.DataFrame:
    initialize_registry(db_path)
    with get_connection(db_path) as conn:
        df = pd.read_sql_query(
            """
            SELECT event_ts, action, patient_id, actor, details
            FROM patient_audit_log
            ORDER BY id DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )
    return df

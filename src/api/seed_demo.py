from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Support direct file execution by making project root importable.
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.api.registry import upsert_patient
from src.utils import load_config

"""Seed script that inserts a few demo patients into the local registry."""


def seed_patients(db_path: str) -> None:
    # Small starter set with mixed risk characteristics.
    demo_patients = [
        {
            "patient_id": "HK-0001",
            "race": "Asian",
            "gender": "Female",
            "age_midpoint": 74,
            "admission_type_id": "1",
            "discharge_disposition_id": "1",
            "admission_source_id": "7",
            "time_in_hospital": 6,
            "num_lab_procedures": 62,
            "num_procedures": 1,
            "num_medications": 18,
            "number_outpatient": 2,
            "number_emergency": 1,
            "number_inpatient": 2,
            "number_diagnoses": 10,
            "A1Cresult": ">8",
            "max_glu_serum": ">300",
            "insulin": "Up",
            "change": "Ch",
            "diabetesMed": "Yes",
            "flag_diabetes": 1,
            "flag_heart_failure": 1,
            "flag_kidney_disease": 0,
            "flag_copd": 0,
        },
        {
            "patient_id": "HK-0002",
            "race": "Asian",
            "gender": "Male",
            "age_midpoint": 58,
            "admission_type_id": "2",
            "discharge_disposition_id": "1",
            "admission_source_id": "2",
            "time_in_hospital": 3,
            "num_lab_procedures": 38,
            "num_procedures": 0,
            "num_medications": 11,
            "number_outpatient": 1,
            "number_emergency": 0,
            "number_inpatient": 0,
            "number_diagnoses": 7,
            "A1Cresult": "Norm",
            "max_glu_serum": "None",
            "insulin": "Steady",
            "change": "No",
            "diabetesMed": "Yes",
            "flag_diabetes": 1,
            "flag_heart_failure": 0,
            "flag_kidney_disease": 0,
            "flag_copd": 0,
        },
        {
            "patient_id": "HK-0003",
            "race": "Asian",
            "gender": "Female",
            "age_midpoint": 81,
            "admission_type_id": "1",
            "discharge_disposition_id": "3",
            "admission_source_id": "4",
            "time_in_hospital": 9,
            "num_lab_procedures": 77,
            "num_procedures": 2,
            "num_medications": 24,
            "number_outpatient": 3,
            "number_emergency": 2,
            "number_inpatient": 3,
            "number_diagnoses": 13,
            "A1Cresult": ">7",
            "max_glu_serum": ">200",
            "insulin": "Down",
            "change": "Ch",
            "diabetesMed": "Yes",
            "flag_diabetes": 1,
            "flag_heart_failure": 1,
            "flag_kidney_disease": 1,
            "flag_copd": 1,
        },
    ]

    for patient in demo_patients:
        upsert_patient(db_path, patient)

    print(f"Seeded {len(demo_patients)} demo patients into {db_path}")


def main(config_path: str) -> None:
    # Use shared config so db path stays consistent with dashboard.
    config = load_config(config_path)
    db_path = config["app"]["database_path"]
    seed_patients(db_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed SQLite registry with demo patients.")
    parser.add_argument("--config", default="config/project_config.yaml", help="Path to project config")
    args = parser.parse_args()
    main(args.config)

# Hospital Readmission Prediction for Hong Kong Public Hospitals

## 1. Project Goal (Simple Version)

This project is a proof-of-concept AI system that predicts whether a chronic-disease patient may be readmitted within 30 days.

The main use case is to help care teams prioritize follow-up actions early, for example:
1. Follow-up phone call
2. Home visit planning
3. Telehealth scheduling

This is designed from a Year-1 Computer Engineering student perspective:
1. Keep architecture practical and understandable
2. Use clear scripts in sequence
3. Prefer reproducible outputs over very advanced optimization

## 2. Why This Matters in Hong Kong Context

Hong Kong Hospital Authority has active digital-health and analytics initiatives. This prototype aligns with that direction by showing:
1. How readmission risk scoring can be done using EHR-like features
2. How local assumptions can be included through calibration
3. How scores can be displayed in an admin dashboard workflow

## 3. What Is Implemented

1. End-to-end data pipeline
2. Three model families for comparison
3. Probability calibration for Hong Kong prevalence assumptions
4. Local patient admin panel (add, update, delete)
5. Batch scoring and risk visualization
6. Suggested intervention text for high-risk patients

## 4. Full Workflow (Run Order)

Run scripts in this exact order:

1. Download external datasets
2. Prepare cleaned modeling dataset
3. Train and compare models
4. (Optional) Seed demo patients
5. Launch dashboard

Commands:

```powershell
python -m src.data.download_datasets
python -m src.data.prepare_dataset
python -m src.models.train_model
python -m src.api.seed_demo
streamlit run dashboard/app.py
```

## 5. Repository Structure (Clear Map)

1. Data stage
2. Model stage
3. App/API stage
4. Dashboard stage
5. Config and artifacts

Detailed map:
1. [src/data/download_datasets.py](src/data/download_datasets.py)
2. [src/data/prepare_dataset.py](src/data/prepare_dataset.py)
3. [src/models/calibration.py](src/models/calibration.py)
4. [src/models/train_model.py](src/models/train_model.py)
5. [src/models/inference.py](src/models/inference.py)
6. [src/api/registry.py](src/api/registry.py)
7. [src/api/seed_demo.py](src/api/seed_demo.py)
8. [dashboard/app.py](dashboard/app.py)
9. [config/project_config.yaml](config/project_config.yaml)
10. [config/hk_health_stats.csv](config/hk_health_stats.csv)

## 6. Code Explanation and Possible Improvements (Per File)

### 6.1 [src/data/download_datasets.py](src/data/download_datasets.py)

What it does:
1. Downloads UCI diabetes dataset zip
2. Extracts CSV into local folder
3. Copies HK stats snapshot into external data folder

Possible improvements:
1. Add retry/backoff on internet failure
2. Verify file checksum
3. Log source metadata (URL, download date, size)

### 6.2 [src/data/prepare_dataset.py](src/data/prepare_dataset.py)

What it does:
1. Cleans missing markers
2. Builds binary target for <30 day readmission
3. Creates simple condition flags from diagnosis prefixes
4. Exports model-ready table

Possible improvements:
1. Add unit tests for feature rules
2. Use stronger ICD mapping logic (beyond prefix matching)
3. Add automatic data quality report

### 6.3 [src/models/calibration.py](src/models/calibration.py)

What it does:
1. Converts probability to log-odds and back
2. Applies prevalence-shift calibration

Possible improvements:
1. Compare with isotonic calibration
2. Add reliability curve evaluation

### 6.4 [src/models/train_model.py](src/models/train_model.py)

What it does:
1. Trains Logistic Regression, XGBoost, and MLP
2. Evaluates ROC-AUC and PR-AUC
3. Saves best model and metadata

Possible improvements:
1. Add cross-validation and hyperparameter search
2. Add model version tracking
3. Add fairness evaluation by subgroups

### 6.5 [src/models/inference.py](src/models/inference.py)

What it does:
1. Loads trained model
2. Scores new patient rows
3. Applies calibration and threshold label

Possible improvements:
1. Validate input schema and data types before scoring
2. Add confidence intervals
3. Add request-level logs for audit

### 6.6 [src/api/registry.py](src/api/registry.py)

What it does:
1. Creates local SQLite patient table
2. Supports upsert, delete, list

Possible improvements:
1. Add timestamp/audit columns
2. Add stronger validation constraints
3. Migrate to SQLAlchemy for larger projects

### 6.7 [src/api/seed_demo.py](src/api/seed_demo.py)

What it does:
1. Inserts demo patients for faster testing

Possible improvements:
1. Load sample data from CSV/JSON
2. Generate synthetic cohorts for stress testing

### 6.8 [dashboard/app.py](dashboard/app.py)

What it does:
1. GUI for add/update/delete patients
2. One-click scoring for all records
3. Risk chart and action suggestions

App flow notes (from dashboard logic):
1. Uses local SQLite registry as admin data source.
2. Loads saved model artifact and metadata before scoring.
3. Scores all current records and labels risk as HIGH/LOW.
4. Shows intervention suggestions for high-risk patients.

Possible improvements:
1. Add authentication and roles
2. Add patient search/filter tools
3. Save historical score snapshots for trend view
4. Add export button for scored results (CSV for reporting)
5. Add basic audit log for admin actions (add/update/delete/score)
6. Add clearer empty-state guidance when model artifacts are missing

## 6.9 Module Improvement Summary (Quick View)

This section summarizes key upgrades you can present module-by-module.

1. [src/utils.py](src/utils.py)
	- Add config schema checks and friendly validation messages.
	- Support environment variable overrides for deployment.
2. [src/data/download_datasets.py](src/data/download_datasets.py)
	- Add download retries and checksum validation.
	- Log data source metadata for reproducibility.
3. [src/data/prepare_dataset.py](src/data/prepare_dataset.py)
	- Add unit tests for feature engineering.
	- Replace prefix-based diagnosis flags with stronger ICD mapping.
	- Add automated data quality profiling.
4. [src/models/calibration.py](src/models/calibration.py)
	- Compare prevalence-shift with isotonic/Platt calibration.
	- Add calibration diagnostics (reliability plots).
5. [src/models/train_model.py](src/models/train_model.py)
	- Add cross-validation and hyperparameter tuning.
	- Track experiments and model versions.
	- Add fairness metrics by subgroup.
6. [src/models/inference.py](src/models/inference.py)
	- Add strict input schema checks.
	- Add scoring logs and optional uncertainty estimates.
7. [src/api/registry.py](src/api/registry.py)
	- Add audit columns (`created_at`, `updated_at`).
	- Add stronger input constraints and validation.
	- Optionally migrate to ORM for maintainability.
8. [src/api/seed_demo.py](src/api/seed_demo.py)
	- Move sample profiles to editable CSV/JSON.
	- Add synthetic bulk generation options.
9. [dashboard/app.py](dashboard/app.py)
	- Add authentication/role-based access.
	- Add filter/search/export tools.
	- Save scoring history and admin action logs.

## 7. Data Sources Used

1. UCI Diabetes 130-US hospitals dataset
2. HK-specific assumption table in [config/hk_health_stats.csv](config/hk_health_stats.csv)

Important note:
1. This prototype does not use real HA patient-level private data.
2. HK values are used for calibration assumptions only.

## 8. Generated Outputs

After running the pipeline, key files are:
1. [data/external/diabetic_data.csv](data/external/diabetic_data.csv)
2. [data/processed/readmission_model_dataset.csv](data/processed/readmission_model_dataset.csv)
3. [artifacts/best_model.joblib](artifacts/best_model.joblib)
4. [artifacts/model_metadata.yaml](artifacts/model_metadata.yaml)
5. [artifacts/model_comparison.csv](artifacts/model_comparison.csv)
6. [artifacts/patients.db](artifacts/patients.db)

## 9. Semester Timeline 

### Week 1-2: Planning and Setup
1. Define project scope and success criteria
2. Collect candidate datasets and identify limitations
3. Build repo skeleton and environment

### Week 3-4: Data Pipeline
1. Implement dataset download and cleaning
2. Build feature engineering rules
3. Validate dataset quality and class balance

### Week 5-6: Baseline Modeling
1. Train baseline model (logistic regression)
2. Add tree model (XGBoost)
3. Compare metrics and store outputs

### Week 7: Calibration and Interpretation
1. Add prevalence-shift calibration for HK scenario
2. Review which features are likely high-impact

### Week 8-9: Dashboard and CRUD
1. Build Streamlit admin interface
2. Add patient CRUD and scoring button
3. Add risk charts and triage text

### Week 10: Testing and Debugging
1. Run full pipeline end-to-end from clean folder
2. Fix edge cases and path/config issues

### Week 11: Documentation and Report
1. Finalize README and technical explanation
2. Add assumptions, risks, and limitations
3. Prepare demo script and screenshots

### Week 12: Presentation and Reflection
1. Present system architecture and results
2. Discuss what worked and what to improve in next version

## 10. Setup Instructions

1. Create environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run pipeline and app using commands in Section 4.

## 11. Limitations

1. Public sample data is not identical to Hong Kong hospital data.
2. Feature engineering is simplified for educational prototyping.
3. No clinical validation, fairness audit, or security hardening yet.

## 12. Overall Improvement Roadmap

1. Data: secure collaboration with local clinical teams for realistic feature schema
2. Modeling: tune models and add explainability methods (for example SHAP)
3. Operations: convert pipeline into scheduled jobs and API services
4. Product: integrate with clinician workflow and patient communication channels
5. Governance: add privacy, fairness, and model monitoring controls

## 13. Disclaimer

1. This repository is an academic prototype.
2. It is not a clinical decision support system for real patient care.
3. Real deployment would require formal validation, governance, and hospital approval.

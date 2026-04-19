# Hospital Readmission Predictor - Diabetes

## 1. Project Goal (Simple Version)

This project is an academic prototype that estimates whether a diabetes-related patient may be readmitted within 30 days.

The practical goal is to help care teams prioritize follow-up earlier, for example:

- follow-up phone calls
- home-visit planning
- telehealth scheduling

This was built from a Year-1 Computer Engineering perspective, so the design choices are intentionally practical:

- keep the architecture understandable
- use a clear script-by-script workflow
- prioritize reproducible outputs over heavy optimization

## Quickstart (Fastest Way)

Use this if you want to run the project with minimal setup.

1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Build artifacts (download, prepare, train)

```powershell
python -m src.data.download_datasets
python -m src.data.prepare_dataset
python -m src.models.train_model
```

3. Seed sample patients (optional but recommended for first run)

```powershell
python -m src.api.seed_demo
```

4. Launch the dashboard

```powershell
python run_dashboard.py
```

5. Use the dashboard
- open the local Streamlit URL shown in terminal (typically http://localhost:8501)
- go to "Patient Registry" to review seeded records
- use "Risk Scoring" to run predictions and view LOW/MEDIUM/HIGH triage bands
- use filters and CSV export for quick cohort review

## 2. Why This Matters in the Hong Kong Context

Hong Kong Hospital Authority has been pushing digital-health and analytics initiatives. This prototype shows how a small team can:

- build a readmission risk workflow with EHR-like features
- apply local assumptions using calibration
- surface scores in an admin-friendly dashboard flow

## 3. What Is Implemented

- end-to-end data pipeline
- three model families for side-by-side comparison
- prevalence-based calibration for Hong Kong assumptions
- local patient admin panel (add, update, delete)
- batch scoring and risk visualization
- intervention suggestions for high-risk patients
- resilient dataset download with retries and metadata logging
- automated data-quality report generation during preprocessing
- richer feature engineering (utilization and acuity features)
- strict inference schema validation with feature ordering enforcement
- three-tier risk bands (LOW/MEDIUM/HIGH) for operational triage
- threshold governance with recall floor and precision-focused selection
- patient registry audit logging (create/update/delete with actor + timestamp)
- dashboard cohort filters, CSV export, model card view, audit trail tab, and feature-driver chart

## 3.1 Current Benchmark (Latest Verified Run)

Latest single-run training artifact summary:

- best model: logistic_regression
- threshold strategy: maximize validation F1 with precision/recall guardrails
- calibrated deployment threshold: 0.175
- held-out ROC-AUC: 0.6760
- held-out PR-AUC: 0.2168
- held-out recall: 0.5407
- held-out precision: 0.1884
- held-out F1: 0.2795

Threshold policy effect vs base threshold 0.30:

- tuned threshold improves precision from 0.1134 to 0.1884
- tuned threshold improves F1 from 0.2037 to 0.2795
- tuned threshold remains above the recall floor at 0.5407

Candidate comparison from artifacts:

- logistic_regression: val PR-AUC 0.2059, val F1@opt-threshold 0.2651 (selected)
- random_forest: val PR-AUC 0.2097, val F1@opt-threshold 0.2606

## 3.2 Current Issues and Risks (Living List)

This section is the single source of truth for current model and product issues.

1. Class imbalance remains severe
- issue: positive readmission rate is about 11.16%.
- impact: false positives rise quickly when recall is increased.
- evidence: `artifacts/model_metadata.yaml` and `data/processed/data_quality_report.csv`.

2. High missingness in key lab predictors
- issue: `A1Cresult` missing ~83.3%, `max_glu_serum` missing ~94.7%.
- impact: model often predicts without strong glycemic signal.
- evidence: `data/processed/data_quality_report.csv`.

3. Threshold tradeoff is unavoidable at current signal quality
- issue: lower threshold improves F1 and recall but reduces precision versus stricter thresholds.
- impact: operational burden from additional false alerts.
- evidence: `artifacts/model_metadata.yaml` under `test_metrics` and `test_metrics_base_threshold`.

4. Moderate ranking quality only
- issue: PR-AUC is ~0.2168, better than prevalence baseline but still not strong.
- impact: limited ceiling for precision at acceptable recall.
- evidence: `artifacts/model_metadata.yaml`.

5. External validity and deployment gap
- issue: dataset is U.S.-origin public data with HK prevalence adjustment, not true local patient-level EHR.
- impact: possible performance drift and calibration mismatch in real deployment.

6. Responsible-AI and governance gaps still open
- issue: no subgroup fairness dashboard, no calibration drift monitor, no production model monitoring loop.
- impact: harder to defend safety and governance quality in real settings.

## 3.3 Future Improvements and Roadmap (Living List)

This section is the single source of truth for planned improvements.

Immediate (next iteration)

1. Precision-focused operating point pack
- run controlled threshold profiles (high-precision, balanced, high-recall) and publish a table for stakeholder choice.
2. Missingness-aware feature upgrades
- add explicit missingness indicator features for high-missing clinical fields.
3. Probability reliability reporting
- generate reliability bins and expected calibration error in artifacts.

Short-term

1. Better class-imbalance handling
- test focal-loss-style approximations or cost-sensitive tuning via sample weighting and class-specific thresholds.
2. Feature engineering refresh
- add interaction terms and temporal-utilization proxies from available visit history fields.
3. Better model search coverage
- include calibrated gradient boosting variants when runtime allows.

Medium-term

1. Subgroup fairness diagnostics
- evaluate precision/recall/FPR by age bands, sex, and race categories in validation artifacts.
2. Drift and monitoring scaffolding
- add data drift checks and score distribution monitoring across runs.
3. Error-analysis notebook
- identify top false-positive and false-negative cohorts with actionable feature patterns.

Long-term

1. Local data alignment
- map to HK-specific coding and clinical workflows, then retrain with local governance approval.
2. Productionization
- move from script pipeline to scheduled jobs + API service with model registry.
3. Access control and audit hardening
- role-based access for dashboard actions and signed audit events.

## 4. Full Workflow (Run Order)

Run the scripts in this order:

1. Download external datasets
2. Prepare the cleaned modeling dataset
3. Train and compare models
4. (Optional) Seed demo patients
5. Launch the dashboard

Commands:

```powershell
python -m src.data.download_datasets
python -m src.data.prepare_dataset
python -m src.models.train_model
python -m src.api.seed_demo
streamlit run dashboard/app.py
```

## 5. Repository Structure (Clear Map)

High-level stages:

- data
- modeling
- app/API
- dashboard
- config and artifacts

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

## 6. Code Explanation and Upgrades (Per File)

### 6.1 [src/data/download_datasets.py](src/data/download_datasets.py)

What it does:

- downloads the UCI diabetes dataset archive
- extracts the CSV into the local external-data folder
- saves an HK stats snapshot into the same run context

Possible improvements:

- add retry/backoff handling for unstable network conditions
- verify checksums for downloaded files
- log source metadata (URL, download date, file size)

### 6.2 [src/data/prepare_dataset.py](src/data/prepare_dataset.py)

What it does:

- cleans missing markers
- builds the binary target for readmission within 30 days
- creates basic condition flags from diagnosis-code prefixes
- exports a model-ready table

Possible improvements:

- add unit tests for feature rules
- replace prefix matching with stronger ICD mapping logic
- generate an automatic data-quality report

### 6.3 [src/models/calibration.py](src/models/calibration.py)

What it does:

- converts probabilities to log-odds and back
- applies prevalence-shift calibration

Possible improvements:

- compare against isotonic calibration
- add reliability-curve evaluation

### 6.4 [src/models/train_model.py](src/models/train_model.py)

What it does:

- trains Logistic Regression, XGBoost, and MLP
- evaluates ROC-AUC and PR-AUC
- saves the best model and metadata

Possible improvements:

- add cross-validation and hyperparameter search
- add model version tracking
- add subgroup fairness evaluation

### 6.5 [src/models/inference.py](src/models/inference.py)

What it does:

- loads trained model artifacts
- scores new patient rows
- applies calibration and threshold-based labels

Possible improvements:

- validate input schema and dtypes before scoring
- add confidence intervals
- add request-level audit logs

### 6.6 [src/api/registry.py](src/api/registry.py)

What it does:

- creates the local SQLite patient table
- supports upsert, delete, and list operations

Possible improvements:

- add timestamp and audit columns
- add stronger validation constraints
- migrate to SQLAlchemy for larger-scale maintainability

### 6.7 [src/api/seed_demo.py](src/api/seed_demo.py)

What it does:

- inserts demo patients for quicker testing

Possible improvements:

- load sample profiles from CSV/JSON
- generate synthetic cohorts for stress testing

### 6.8 [dashboard/app.py](dashboard/app.py)

What it does:

- provides GUI actions to add, update, and delete patients
- supports one-click scoring for current records
- shows risk charts and intervention suggestions

App flow notes:

1. Uses local SQLite registry as the admin data source.
2. Loads model artifact and metadata before scoring.
3. Scores records and labels risk as HIGH/LOW.
4. Displays intervention suggestions for high-risk patients.

Possible improvements:

- add authentication and role control
- add patient search/filter tools
- save historical score snapshots for trend views
- add export for scored results (CSV)
- add basic admin audit logs (add/update/delete/score)
- improve empty-state guidance when model artifacts are missing

### 6.9 Module Improvement Summary (Quick View)

This section summarizes useful upgrades module by module.

1. [src/utils.py](src/utils.py)
	- add config schema checks and clearer validation messages
	- support environment-variable overrides for deployment
2. [src/data/download_datasets.py](src/data/download_datasets.py)
	- add retries and checksum validation
	- log source metadata for reproducibility
3. [src/data/prepare_dataset.py](src/data/prepare_dataset.py)
	- add unit tests for feature engineering
	- replace prefix rules with stronger ICD mapping
	- add automated data-quality profiling
4. [src/models/calibration.py](src/models/calibration.py)
	- compare prevalence-shift with isotonic/Platt calibration
	- add calibration diagnostics (reliability plots)
5. [src/models/train_model.py](src/models/train_model.py)
	- add cross-validation and hyperparameter tuning
	- track experiments and model versions
	- add subgroup fairness metrics
6. [src/models/inference.py](src/models/inference.py)
	- add stricter input schema checks
	- add scoring logs and optional uncertainty estimates
7. [src/api/registry.py](src/api/registry.py)
	- add audit columns (created_at, updated_at)
	- add stronger input constraints
	- optionally migrate to ORM for maintainability
8. [src/api/seed_demo.py](src/api/seed_demo.py)
	- move sample profiles to editable CSV/JSON
	- add synthetic bulk generation options
9. [dashboard/app.py](dashboard/app.py)
	- add authentication/role-based access
	- add filter/search/export tools
	- save scoring history and admin action logs

## 7. Data Sources Used

- UCI Diabetes 130-US hospitals dataset
- HK-specific assumption table in [config/hk_health_stats.csv](config/hk_health_stats.csv)

Important note:

- this prototype does not use real HA patient-level private data
- HK values are used for calibration assumptions only

## 8. Generated Outputs

After running the pipeline, key files include:

1. [data/external/diabetic_data.csv](data/external/diabetic_data.csv)
2. [data/processed/readmission_model_dataset.csv](data/processed/readmission_model_dataset.csv)
3. [artifacts/best_model.joblib](artifacts/best_model.joblib)
4. [artifacts/model_metadata.yaml](artifacts/model_metadata.yaml)
5. [artifacts/model_comparison.csv](artifacts/model_comparison.csv)
6. [artifacts/patients.db](artifacts/patients.db)

## 9. Semester Timeline

### Week 1-2: Planning and Setup

- define project scope and success criteria
- collect candidate datasets and identify limitations
- build repo skeleton and environment

### Week 3-4: Data Pipeline

- implement dataset download and cleaning
- build feature-engineering rules
- validate dataset quality and class balance

### Week 5-6: Baseline Modeling

- train baseline model (logistic regression)
- add tree model (XGBoost)
- compare metrics and store outputs

### Week 7: Calibration and Interpretation

- add prevalence-shift calibration for HK scenario
- review likely high-impact features

### Week 8-9: Dashboard and CRUD

- build Streamlit admin interface
- add patient CRUD and scoring action
- add risk charts and triage text

### Week 10: Testing and Debugging

- run the full pipeline end-to-end from a clean folder
- fix edge cases and path/config issues

### Week 11: Documentation and Report

- finalize README and technical explanation
- add assumptions, risks, and limitations
- prepare demo script and screenshots

### Week 12: Presentation and Reflection

- present system architecture and results
- discuss what worked and what to improve next

## 10. Setup Instructions

Create an environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then run the pipeline and app using the commands in Section 4.

## 11. Limitations

- public sample data is not identical to Hong Kong hospital data
- feature engineering is simplified for educational prototyping
- no clinical validation, fairness audit, or security hardening yet

## 12. Overall Improvement Roadmap

1. Data: collaborate with local clinical teams for realistic feature schema
2. Modeling: tune models and add explainability methods (for example, SHAP)
3. Operations: convert pipeline into scheduled jobs and API services
4. Product: integrate with clinician workflow and patient communication channels
5. Governance: add privacy, fairness, and model-monitoring controls

## 13. Disclaimer

- this repository is an academic prototype
- it is not a clinical decision support system for real patient care
- real deployment would require formal validation, governance, and hospital approval

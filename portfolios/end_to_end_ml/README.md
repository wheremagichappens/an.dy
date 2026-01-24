# Predicting Customer Disengagement & Evaluating Targeted Interventions

This repo is a small, **end-to-end applied ML system**: ETL → labeling → training → scoring → intervention evaluation → monitoring.
It’s designed to look and feel like a production-style Applied Scientist project (batch pipeline + artifacts + drift checks).

## What this does
- Builds customer-level behavioral features from transaction logs
- Defines a leakage-safe churn label (no purchase in the next 30 days after a snapshot date)
- Trains baseline models (Logistic Regression + Gradient Boosting)
- Scores customers and creates a “high-risk” segment (top 20%)
- Evaluates intervention ROI + a holdout-style causal lift estimate
- Monitors feature drift + score stability (PSI) and flags retraining

## Data
This project expects a CSV at:
- `data/raw/online_retail.csv`

It’s compatible with UCI Online Retail / Online Retail II style columns:
- `CustomerID`, `InvoiceDate`, `InvoiceNo`, `Quantity`, `UnitPrice`, `StockCode`, `Country`

> Note: the repo does **not** include the dataset (license/size). Drop your CSV into `data/raw/online_retail.csv`.

## Quickstart

1) Create a venv and install deps:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Put your dataset at `data/raw/online_retail.csv`

3) Run the full pipeline:
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

Or run stage-by-stage:
```bash
make all
```

## Outputs
- Processed tables:
  - `data/processed/customer_features.csv`
  - `data/processed/customer_features_labeled.csv`
  - `data/processed/scored_customers.csv`
- Model artifacts:
  - `artifacts/models/logreg.joblib`
  - `artifacts/models/gbm.joblib`
- Metrics:
  - `artifacts/metrics/logreg.json`
  - `artifacts/metrics/gbm.json`
  - `artifacts/metrics/intervention_roi_summary.csv`
  - `artifacts/metrics/holdout_causal_check.csv`
- Monitoring:
  - `artifacts/monitoring/feature_drift_report.csv`
  - `artifacts/monitoring/score_stability_report.csv`

## Repo structure
```text
etl/            # Data preparation
labels/         # Churn labeling
models/         # Training & scoring
evaluation/     # ROI & causal checks
monitoring/     # Drift & stability monitoring
artifacts/      # Saved models & metrics
data/           # Raw + processed data
```

## Notes / next upgrades
- Replace simulated holdout with real experiment assignment + logged exposures
- Add time-based split (train on earlier, validate on later)
- Add retraining scheduler (cron / Airflow) and artifact versioning


## Causal inference (Uber-style)
This repo includes:
- **Holdout causal check** (high-risk segment) in `evaluation/causal_checks.py`
- **CUPED variance reduction** in `causal/cuped_analysis.py`
- **Difference-in-Differences** using a country×week panel in `causal/diff_in_diff.py`
- **Parallel trends plot** in `causal/parallel_trends_plot.py`

## Dataset
Place the UCI Online Retail dataset (converted to CSV) at:
- `data/raw/online_retail.csv`

The raw dataset is gitignored by default.

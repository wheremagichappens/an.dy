
# Customer Churn Intervention with Causal Inference  
**End-to-End Applied ML, Experimentation, and Incrementality (Production-Style)**

---

## Executive Summary

This repository implements a **production-style churn intervention pipeline** designed to answer a core business question:

> **If we intervene on high-risk customers, does it *causally* reduce churn — and is the impact worth the cost?**

Unlike typical churn projects that stop at prediction, this system integrates:
- Predictive churn modeling (who is at risk)
- Targeting and intervention logic (who to act on)
- Causal inference (did the action *cause* improvement?)
- ROI-based decision-making (should we roll out?)
- Monitoring for deployment readiness

The overall design mirrors **marketplace and growth experimentation workflows** used at companies such as **Uber, DoorDash, and Thumbtack**, where decisions must be **incremental and causal**, not correlational.

---

## Repository Structure (Source of Truth)

```text
customer-churn-intervention-ml/
├── README.md
├── requirements.txt
├── run_pipeline.sh
├── Makefile
│
├── data/
│   ├── raw/
│   │   └── online_retail.csv
│   └── processed/
│       ├── customer_features_labeled.csv
│       └── country_week_panel.csv
│
├── etl/
│   └── build_customer_table.py
│
├── labels/
│   └── build_churn_label.py
│
├── models/
│   ├── train_model.py
│   └── score_customers.py
│
├── evaluation/
│   ├── intervention_analysis.py
│   └── causal_checks.py
│
├── causal/
│   ├── cuped_analysis.py
│   ├── build_country_panel.py
│   ├── diff_in_diff.py
│   └── parallel_trends_plot.py
│
├── monitoring/
│   ├── data_drift.py
│   └── score_stability.py
│
└── artifacts/
    ├── models/
    ├── metrics/
    └── monitoring/
```

This README references **only files that actually exist in the project**.

---

## End-to-End Data & Decision Flow

```text
Raw transactions
(data/raw/online_retail.csv)
        │
        ▼
ETL & feature engineering
(etl/build_customer_table.py)
        │
        ▼
Customer features + pre-period metrics
(data/processed/customer_features_labeled.csv)
        │
        ▼
Churn labeling (leakage-safe)
(labels/build_churn_label.py)
        │
        ▼
Churn risk model
(models/train_model.py)
        │
        ▼
Risk scoring
(models/score_customers.py)
        │
        ▼
Targeting + holdout assignment
(evaluation/causal_checks.py)
        │
        ▼
Causal inference layer
├─ Holdout lift (evaluation/causal_checks.py)
├─ CUPED variance reduction (causal/cuped_analysis.py)
└─ DiD, market-level (causal/diff_in_diff.py)
        │
        ▼
ROI-informed decision + monitoring
(monitoring/data_drift.py,
 monitoring/score_stability.py)
```

---

## Dataset

**Source:** UCI Machine Learning Repository — *Online Retail Dataset*

**Why this dataset is appropriate**
- Real transactional data (not synthetic)
- Time-stamped events enable pre/post analysis
- Multiple aggregation levels (customer, country)
- Supports both randomized and quasi-experimental designs

**Placement (not committed to Git):**
```text
data/raw/online_retail.csv
```

---

## Churn Modeling (Prediction ≠ Decision)

Customer-level features include:
- Recency (days since last purchase)
- Frequency (purchase count)
- Monetary value (spend)
- Pre-period behavioral metrics

Churn labels are constructed in a **forward-looking, leakage-safe** manner:
```text
labels/build_churn_label.py
```

A baseline churn model is trained and scored:
```text
models/train_model.py
models/score_customers.py
```

**Important:**  
The churn model is used **only for targeting** — not for claiming business impact.

---

## Causal Inference: Measuring Incrementality

### 1. Holdout-Based Incrementality (User-Level)

High-risk customers are split into:
- **Treatment**: receive intervention
- **Control (holdout)**: receive nothing

Incremental effect:
```
Retention_treatment − Retention_control
```

Implemented in:
```text
evaluation/causal_checks.py
```

This removes bias from:
- Regression to the mean
- Natural recovery
- Seasonality

---

### 2. CUPED (Variance Reduction)

To detect lift faster and more reliably, the pipeline applies **CUPED**, conditioning on pre-period behavior.

Implemented in:
```text
causal/cuped_analysis.py
```

This mirrors variance-reduction techniques used in large-scale experimentation platforms.

---

### 3. Difference-in-Differences (Market-Level)

When user-level randomization is not feasible (e.g., market rollouts), the pipeline switches to DiD:

Steps:
1. Build a country × week panel  
   (`causal/build_country_panel.py`)
2. Estimate DiD with fixed effects  
   (`causal/diff_in_diff.py`)
3. Validate assumptions via parallel trends  
   (`causal/parallel_trends_plot.py`)

---

## Outputs

Key artifacts produced by the pipeline:

```text
artifacts/
├── models/
│   └── churn_model.pkl
├── metrics/
│   ├── cuped_results.csv
│   ├── did_results.csv
│   └── parallel_trends_plot.png
└── monitoring/
    ├── feature_drift.csv
    └── score_stability.csv
```

These outputs are designed to support **go / no-go rollout decisions**, not just model evaluation.

---

## Assumptions & Limitations

- The dataset does not include a real intervention flag
- Treatment assignment is simulated for demonstration
- Causal estimates are illustrative, not production claims

The emphasis is on **correct experimental design, assumptions, and decision logic**.

---

## How to Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Place dataset
data/raw/online_retail.csv

chmod +x run_pipeline.sh
./run_pipeline.sh
```

---

## Intended Audience

- Marketplace / Growth Data Scientists
- Experimentation & Causal Inference roles
- Applied ML portfolios focused on business impact

---

## Why This Project Matters

This project demonstrates the difference between:
- **Predicting risk**
- **Measuring causal impact**
- **Making defensible business decisions**

That distinction is critical in real-world data science.
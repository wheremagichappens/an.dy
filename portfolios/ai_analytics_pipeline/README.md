# AI-Powered Campaign Evaluation Pipeline

A production-style Python system for evaluating marketing and product experiments in **non-randomized settings**, combining causal inference methods with scalable analytics workflows.

This project demonstrates how to move from **manual analysis → reusable decision system**, and serves as a foundation for **AI-driven, self-serve analytics**.

---

## Motivation

In real-world product and marketing scenarios, randomized A/B testing is often not feasible due to:
- targeted campaigns  
- operational constraints  
- geographic or segment-based rollouts  

Naive before/after comparisons can lead to **biased and inflated estimates of impact**.

This project addresses that by building a **robust, reusable evaluation system** that produces **trustworthy causal estimates at scale**.

---

## What This System Does

### 1. Feature Engineering
Builds user-level features from raw behavioral data:
- trip frequency and recency  
- engagement patterns  
- station usage diversity  
- location-based features  

---

### 2. Propensity Score Matching (PSM)
- Estimates treatment probability using logistic regression  
- Matches treated and control users via nearest neighbors  
- Applies configurable **caliper thresholds**  
- Creates statistically comparable groups  

---

### 3. Balance Diagnostics
- Computes **Standardized Mean Differences (SMD)**  
- Validates covariate balance  
- Flags imbalance issues for debugging  

---

### 4. Difference-in-Differences (DiD)
- Estimates **incremental treatment effect**  
- Controls for time trends  
- Uses **clustered standard errors (user-level)**  

---

### 5. Assumption Validation
- Pre-treatment trend checks (parallel trends)  
- Sensitivity analysis across segments and thresholds  

---

### 6. Automated Reporting
- Generates structured outputs:
  - treatment effect  
  - confidence intervals  
  - statistical significance  
- Produces **business-ready summaries** for stakeholders  

---

## AI-for-Analytics Vision

This system is intentionally designed as a **foundation for AI-powered analytics**.

Instead of:
> “Analyst manually builds analysis every time”

We move toward:
> “System generates insights automatically”

### Future Extensions
- Natural language → SQL interface  
- Automated experiment readouts (LLM-generated summaries)  
- Insight recommendation engine (e.g., “which segments to target next”)  
- Conversational analytics interface  

# Campaign Evaluation Pipeline

A production-style Python project that demonstrates a reusable experimentation pipeline for campaign evaluation when randomized A/B testing is not feasible.

It includes:
- mock data generation
- feature engineering
- propensity score modeling and nearest-neighbor matching
- covariate balance diagnostics
- difference-in-differences estimation
- pre-trend checks
- narrative reporting for non-technical stakeholders
- CLI entrypoints
- unit tests

## Project tree

```text
instacart_ai_analytics_pipeline/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── data/
│   └── mock/
│       ├── users.csv
│       ├── trips.csv
│       ├── treatment.csv
│       └── weekly_panel.csv
├── scripts/
│   ├── generate_mock_data.py
│   └── run_pipeline.py
├── src/
│   └── campaign_eval/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── data_generation.py
│       ├── did.py
│       ├── diagnostics.py
│       ├── features.py
│       ├── matching.py
│       ├── pipeline.py
│       ├── reporting.py
│       └── utils/
│           └── logging_utils.py
└── tests/
    └── test_pipeline.py
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Generate mock data:

```bash
python scripts/generate_mock_data.py
```

Run the pipeline:

```bash
python scripts/run_pipeline.py
```

Or via CLI:

```bash
campaign-eval generate-mock-data --output-dir data/mock --n-users 1500
campaign-eval run --data-dir data/mock
```

Run tests:

```bash
pytest -q
```

## What this project demonstrates

This project is meant to be interview-friendly and GitHub-friendly.
It is not presented as a drop-in production system, but it uses production-minded patterns:
- modular package layout
- typed functions and dataclasses
- configuration object
- logging
- validation and error handling
- reproducible mock data
- test coverage for the core path

## AI-for-analytics angle

A practical extension of this project would be to place a natural-language interface on top of the reporting output so stakeholders can:
- ask campaign questions in plain English
- get automated summaries
- receive decision recommendations with guardrails

That aligns with an AI-for-analytics vision: turning complex analysis into a repeatable, self-serve decision system.

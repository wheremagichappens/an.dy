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

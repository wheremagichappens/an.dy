#!/usr/bin/env bash
set -euo pipefail

echo "=== [1/6] ETL: build customer features ==="
python etl/build_customer_table.py

echo "=== [2/6] Labels: build churn label ==="
python labels/build_churn_label.py

echo "=== [3/6] Train: baseline models ==="
python models/train_model.py

echo "=== [4/6] Score: generate risk scores + high-risk segment ==="
python models/score_customers.py

echo "=== [5/6] Evaluation: ROI + holdout causal check ==="
python evaluation/intervention_analysis.py
python evaluation/causal_checks.py

echo "=== [5.5/6] Causal: CUPED + DiD + parallel trends plot ==="
python causal/build_country_panel.py
python causal/diff_in_diff.py
python causal/parallel_trends_plot.py
python causal/cuped_analysis.py

echo "=== [6/6] Monitoring: feature drift + score stability ==="
python monitoring/data_drift.py
python monitoring/score_stability.py

echo "✅ Pipeline complete."

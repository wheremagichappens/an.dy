from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

BASELINE_SCORES_PATH = Path("data/processed/scored_customers.csv")
CURRENT_SCORES_PATH = Path("data/processed/scored_customers.csv")
OUTPUT_PATH = Path("artifacts/monitoring/score_stability_report.csv")

SCORE_COL = "risk_score"
PSI_THRESHOLD = 0.20
N_BINS = 10

def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) < 50 or len(actual) < 50:
        return float("nan")

    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    exp_counts, _ = np.histogram(expected, bins=breakpoints)
    act_counts, _ = np.histogram(actual, bins=breakpoints)

    exp_perc = exp_counts / max(exp_counts.sum(), 1)
    act_perc = act_counts / max(act_counts.sum(), 1)

    exp_perc = np.where(exp_perc == 0, 1e-6, exp_perc)
    act_perc = np.where(act_perc == 0, 1e-6, act_perc)

    return float(np.sum((act_perc - exp_perc) * np.log(act_perc / exp_perc)))

def main():
    base = pd.read_csv(BASELINE_SCORES_PATH)
    curr = pd.read_csv(CURRENT_SCORES_PATH)

    if SCORE_COL not in base.columns or SCORE_COL not in curr.columns:
        raise ValueError(f"Missing {SCORE_COL} in scored files.")

    base_scores = base[SCORE_COL].to_numpy(dtype=float)
    curr_scores = curr[SCORE_COL].to_numpy(dtype=float)

    score_psi = psi(base_scores, curr_scores, bins=N_BINS)
    retrain_flag = (score_psi >= PSI_THRESHOLD) if not np.isnan(score_psi) else False

    report = pd.DataFrame([{
        "score_psi": score_psi,
        "psi_threshold": PSI_THRESHOLD,
        "retrain_flag": retrain_flag,
        "baseline_rows": len(base),
        "current_rows": len(curr),
    }])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_PATH, index=False)

    print("=== Score Stability Report ===")
    print(report.to_string(index=False))
    print(f"Saved: {OUTPUT_PATH}")
    print("Action:", "retrain recommended" if retrain_flag else "no retrain needed")

if __name__ == "__main__":
    main()

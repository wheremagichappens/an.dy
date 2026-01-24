from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

BASELINE_PATH = Path("data/processed/customer_features_labeled.csv")
CURRENT_PATH = Path("data/processed/customer_features_labeled.csv")
OUTPUT_PATH = Path("artifacts/monitoring/feature_drift_report.csv")

TARGET_COL = "churned"
DROP_COLS = ["first_purchase_date", "last_purchase_date", "future_purchase_count"]

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

def prep_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in [TARGET_COL] + DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[numeric_cols]

def main():
    baseline = pd.read_csv(BASELINE_PATH)
    current = pd.read_csv(CURRENT_PATH)

    X_base = prep_numeric(baseline)
    X_curr = prep_numeric(current)

    common_cols = [c for c in X_base.columns if c in X_curr.columns]
    rows = []
    for col in common_cols:
        v_base = X_base[col].to_numpy(dtype=float)
        v_curr = X_curr[col].to_numpy(dtype=float)
        v_psi = psi(v_base, v_curr, bins=N_BINS)
        rows.append({
            "feature": col,
            "psi": v_psi,
            "drift_flag": (v_psi >= PSI_THRESHOLD) if not np.isnan(v_psi) else False
        })

    report = pd.DataFrame(rows).sort_values("psi", ascending=False)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT_PATH, index=False)

    drifted = int(report["drift_flag"].sum())
    print("=== Feature Drift Report ===")
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Features flagged (PSI >= {PSI_THRESHOLD}): {drifted}/{len(report)}")
    print(report.head(10).to_string(index=False))

if __name__ == "__main__":
    main()

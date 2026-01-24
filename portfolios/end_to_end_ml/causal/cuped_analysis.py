from pathlib import Path
import numpy as np
import pandas as pd

IN_PATH = Path("data/processed/customer_features_labeled.csv")
OUT_PATH = Path("artifacts/metrics/cuped_results.csv")

PRE_COL = "purchase_frequency"   # baseline predictor
OUTCOME_CHURN_COL = "churned"    # 1=churn; we'll convert to retention

HOLDOUT_RATE = 0.20
SEED = 42

def assign_holdout(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    r = rng.random(len(df))
    out = df.copy()
    out["group"] = np.where(r < HOLDOUT_RATE, "control", "treatment")
    return out

def cuped_adjust(y: np.ndarray, x: np.ndarray):
    x = x.astype(float)
    y = y.astype(float)
    theta = np.cov(y, x)[0, 1] / np.var(x)
    y_adj = y - theta * (x - x.mean())
    return y_adj, float(theta)

def main():
    df = pd.read_csv(IN_PATH)

    # "Targeted intervention" demo: focus on top 30% most stale users by recency
    df = df.sort_values("recency_days", ascending=False).head(int(len(df) * 0.30)).copy()

    # Create groups
    df = assign_holdout(df)

    # Outcome: retention (1=retained)
    df["retained"] = 1 - df[OUTCOME_CHURN_COL].astype(int)

    # Raw lift
    raw_means = df.groupby("group")["retained"].mean()
    raw_lift = float(raw_means["treatment"] - raw_means["control"])

    # CUPED
    y_adj, theta = cuped_adjust(df["retained"].to_numpy(), df[PRE_COL].to_numpy())
    df["retained_cuped"] = y_adj
    cuped_means = df.groupby("group")["retained_cuped"].mean()
    cuped_lift = float(cuped_means["treatment"] - cuped_means["control"])

    out = pd.DataFrame([{
        "pre_metric": PRE_COL,
        "raw_lift": raw_lift,
        "cuped_lift": cuped_lift,
        "theta": theta,
        "n_treatment": int((df["group"] == "treatment").sum()),
        "n_control": int((df["group"] == "control").sum()),
    }])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print("=== CUPED Results ===")
    print(out.to_string(index=False))
    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()

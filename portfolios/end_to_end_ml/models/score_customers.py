from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

LABELED_DATA_PATH = Path("data/processed/customer_features_labeled.csv")
MODEL_PATH = Path("artifacts/models/gbm.joblib")
OUTPUT_PATH = Path("data/processed/scored_customers.csv")

TARGET_COL = "churned"
DROP_COLS = ["first_purchase_date", "last_purchase_date", "future_purchase_count"]

HIGH_RISK_PERCENTILE = 0.80  # top 20%

def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def prep_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[numeric_cols]

def main():
    df = load_data(LABELED_DATA_PATH)
    model = joblib.load(MODEL_PATH)

    X = prep_features(df)
    risk_score = model.predict_proba(X)[:, 1]

    out = df[["CustomerID"]].copy()
    out["risk_score"] = risk_score

    threshold = np.quantile(out["risk_score"], HIGH_RISK_PERCENTILE)
    out["high_risk"] = (out["risk_score"] >= threshold).astype(int)

    if TARGET_COL in df.columns:
        out["actual_churned"] = df[TARGET_COL].astype(int)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.sort_values("risk_score", ascending=False).to_csv(OUTPUT_PATH, index=False)

    print(f"Saved scored customers to {OUTPUT_PATH}")
    print(f"High-risk threshold (top {int((1-HIGH_RISK_PERCENTILE)*100)}%): {threshold:.4f}")
    print("High-risk count:", out["high_risk"].sum())

    if "actual_churned" in out.columns:
        churn_rate_all = out["actual_churned"].mean()
        churn_rate_high = out.loc[out["high_risk"] == 1, "actual_churned"].mean()
        churn_rate_low = out.loc[out["high_risk"] == 0, "actual_churned"].mean()
        print(f"Churn rate (all): {churn_rate_all:.2%}")
        print(f"Churn rate (high-risk): {churn_rate_high:.2%}")
        print(f"Churn rate (low-risk): {churn_rate_low:.2%}")

if __name__ == "__main__":
    main()

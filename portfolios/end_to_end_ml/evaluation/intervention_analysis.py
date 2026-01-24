from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

SCORED_PATH = Path("data/processed/scored_customers.csv")
CUSTOMER_FEATURES_PATH = Path("data/processed/customer_features.csv")
OUTPUT_PATH = Path("artifacts/metrics/intervention_roi_summary.csv")

OFFER_COST_PER_USER = 2.00
TARGET_HIGH_RISK_ONLY = True
INCREMENTAL_MARGIN = 0.30

def load_scored(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"CustomerID", "risk_score", "high_risk"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in scored file: {missing}")
    return df

def load_customer_revenue(path: Path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "CustomerID" not in df.columns or "total_revenue" not in df.columns:
        return None
    return df[["CustomerID", "total_revenue"]].copy()

def main():
    scored = load_scored(SCORED_PATH)
    revenue_df = load_customer_revenue(CUSTOMER_FEATURES_PATH)

    if revenue_df is not None:
        scored = scored.merge(revenue_df, on="CustomerID", how="left")
        scored["total_revenue"] = scored["total_revenue"].fillna(scored["total_revenue"].median())
        avg_rev_all = float(scored["total_revenue"].mean())
    else:
        scored["total_revenue"] = np.nan
        avg_rev_all = 20.0

    target = scored[scored["high_risk"] == 1].copy() if TARGET_HIGH_RISK_ONLY else scored.copy()
    n_target = len(target)

    avg_rev = float(target["total_revenue"].mean()) if revenue_df is not None else avg_rev_all

    denom = max(avg_rev * INCREMENTAL_MARGIN, 1e-9)
    break_even_lift_rate = OFFER_COST_PER_USER / denom

    lift_rates = [0.01, 0.02, 0.05, 0.10, 0.15]
    rows = []
    for lift in lift_rates:
        incremental_profit_per_user = lift * (avg_rev * INCREMENTAL_MARGIN)
        net_profit_per_user = incremental_profit_per_user - OFFER_COST_PER_USER
        total_net_profit = net_profit_per_user * n_target
        rows.append({
            "target_users": n_target,
            "avg_baseline_revenue": avg_rev,
            "margin_assumption": INCREMENTAL_MARGIN,
            "offer_cost_per_user": OFFER_COST_PER_USER,
            "assumed_lift_rate": lift,
            "incremental_profit_per_user": incremental_profit_per_user,
            "net_profit_per_user": net_profit_per_user,
            "total_net_profit": total_net_profit,
        })

    out = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print("=== ROI / Break-even Summary ===")
    print(f"Target group size: {n_target}")
    print(f"Avg baseline revenue (proxy): {avg_rev:.2f}")
    print(f"Offer cost per user: {OFFER_COST_PER_USER:.2f}")
    print(f"Margin assumption: {INCREMENTAL_MARGIN:.0%}")
    print(f"Break-even lift rate: {break_even_lift_rate:.2%}")
    print(f"Saved scenarios to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

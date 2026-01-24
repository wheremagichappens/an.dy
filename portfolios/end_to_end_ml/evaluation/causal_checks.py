from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

SCORED_PATH = Path("data/processed/scored_customers.csv")
OUTPUT_PATH = Path("artifacts/metrics/holdout_causal_check.csv")

HOLDOUT_RATE = 0.20
RANDOM_SEED = 42
OUTCOME_COL = "actual_churned"  # 1 churned, 0 retained

def load_scored(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"CustomerID", "high_risk"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in scored file: {missing}")
    return df

def assign_holdout(df: pd.DataFrame, holdout_rate: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["rand"] = rng.random(len(df))
    df["group"] = np.where(df["rand"] < holdout_rate, "control", "treatment")
    return df.drop(columns=["rand"])

def srm_check(df: pd.DataFrame) -> dict:
    obs_control = (df["group"] == "control").sum()
    obs_treat = (df["group"] == "treatment").sum()
    total = obs_control + obs_treat
    exp_control = total * HOLDOUT_RATE
    exp_treat = total * (1 - HOLDOUT_RATE)

    chi2 = (obs_control - exp_control) ** 2 / max(exp_control, 1e-9) + (obs_treat - exp_treat) ** 2 / max(exp_treat, 1e-9)
    pval = 1 - stats.chi2.cdf(chi2, df=1)
    return {"srm_chi2": float(chi2), "srm_pvalue": float(pval), "obs_control": int(obs_control), "obs_treatment": int(obs_treat)}

def diff_in_proportions_ci(p1, n1, p2, n2, alpha=0.05):
    se = np.sqrt(p1 * (1 - p1) / max(n1, 1) + p2 * (1 - p2) / max(n2, 1))
    z = stats.norm.ppf(1 - alpha / 2)
    lo = (p1 - p2) - z * se
    hi = (p1 - p2) + z * se
    return float(lo), float(hi), float(se)

def main():
    df = load_scored(SCORED_PATH)
    target = df[df["high_risk"] == 1].copy()
    target = assign_holdout(target, HOLDOUT_RATE, RANDOM_SEED)

    srm = srm_check(target)

    if OUTCOME_COL not in target.columns:
        raise ValueError(
            f"Missing outcome column '{OUTCOME_COL}'. Ensure scored_customers.csv includes actual_churned."
        )

    target["retained"] = 1 - target[OUTCOME_COL].astype(int)
    treat = target[target["group"] == "treatment"]
    control = target[target["group"] == "control"]

    p_treat = treat["retained"].mean()
    p_control = control["retained"].mean()
    lift = p_treat - p_control

    n_treat = len(treat)
    n_control = len(control)

    ci_lo, ci_hi, se = diff_in_proportions_ci(p_treat, n_treat, p_control, n_control)
    z = lift / max(se, 1e-9)
    pval = 2 * (1 - stats.norm.cdf(abs(z)))

    out = pd.DataFrame([{
        "segment": "high_risk",
        "holdout_rate": HOLDOUT_RATE,
        "n_treatment": n_treat,
        "n_control": n_control,
        "retention_treatment": float(p_treat),
        "retention_control": float(p_control),
        "lift_retention": float(lift),
        "ci95_low": ci_lo,
        "ci95_high": ci_hi,
        "z_stat": float(z),
        "p_value": float(pval),
        **srm
    }])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print("=== Holdout Causal Check (High-risk segment) ===")
    print(out.to_string(index=False))
    print(f"Saved results to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

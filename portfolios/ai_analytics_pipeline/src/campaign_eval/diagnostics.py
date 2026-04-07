from __future__ import annotations

import numpy as np
import pandas as pd


def standardized_mean_difference(treated_vals: pd.Series, control_vals: pd.Series) -> float:
    mean_diff = float(treated_vals.mean() - control_vals.mean())
    pooled_std = np.sqrt((treated_vals.var(ddof=1) + control_vals.var(ddof=1)) / 2)
    if pooled_std == 0 or np.isnan(pooled_std):
        return 0.0
    return float(mean_diff / pooled_std)


def balance_table(scored_df: pd.DataFrame, matched_pairs: pd.DataFrame, feature_cols: list[str], user_id_col: str = "user_id") -> pd.DataFrame:
    treated_ids = matched_pairs["treated_user_id"].tolist()
    control_ids = matched_pairs["control_user_id"].tolist()

    matched = scored_df[scored_df[user_id_col].isin(treated_ids + control_ids)].copy()
    rows = []
    for col in feature_cols:
        treated_vals = matched.loc[matched[user_id_col].isin(treated_ids), col]
        control_vals = matched.loc[matched[user_id_col].isin(control_ids), col]
        smd = standardized_mean_difference(treated_vals, control_vals)
        rows.append(
            {
                "feature": col,
                "treated_mean": float(treated_vals.mean()),
                "control_mean": float(control_vals.mean()),
                "smd": smd,
                "balanced_flag": abs(smd) < 0.1,
            }
        )
    return pd.DataFrame(rows).sort_values("smd", key=np.abs, ascending=False)

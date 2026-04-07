from __future__ import annotations

from typing import Any

import pandas as pd
import statsmodels.formula.api as smf


def build_panel_data(
    weekly_df: pd.DataFrame,
    matched_pairs: pd.DataFrame,
    post_period_start: str,
    user_id_col: str = "user_id",
    date_col: str = "week_start",
    post_col: str = "post",
) -> pd.DataFrame:
    keep_ids = set(matched_pairs["treated_user_id"].tolist() + matched_pairs["control_user_id"].tolist())
    panel = weekly_df[weekly_df[user_id_col].isin(keep_ids)].copy()
    panel[date_col] = pd.to_datetime(panel[date_col])
    panel[post_col] = (panel[date_col] >= pd.to_datetime(post_period_start)).astype(int)
    return panel


def run_did(
    panel_df: pd.DataFrame,
    outcome_col: str = "weekly_trips",
    treatment_col: str = "treated",
    post_col: str = "post",
    user_id_col: str = "user_id",
) -> Any:
    formula = f"{outcome_col} ~ {treatment_col} + {post_col} + {treatment_col}:{post_col}"
    return smf.ols(formula=formula, data=panel_df).fit(cov_type="cluster", cov_kwds={"groups": panel_df[user_id_col]})


def run_pretrend_check(
    panel_df: pd.DataFrame,
    outcome_col: str = "weekly_trips",
    treatment_col: str = "treated",
    user_id_col: str = "user_id",
    date_col: str = "week_start",
    post_col: str = "post",
) -> Any:
    pre_df = panel_df[panel_df[post_col] == 0].copy()
    min_date = pre_df[date_col].min()
    pre_df["week_index"] = ((pre_df[date_col] - min_date).dt.days / 7).astype(int)
    formula = f"{outcome_col} ~ {treatment_col} + week_index + {treatment_col}:week_index"
    return smf.ols(formula=formula, data=pre_df).fit(cov_type="cluster", cov_kwds={"groups": pre_df[user_id_col]})

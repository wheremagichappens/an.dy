from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def build_summary(
    did_model: Any,
    pretrend_model: Any,
    matched_pairs: pd.DataFrame,
    treatment_col: str = "treated",
    post_col: str = "post",
) -> Dict[str, float | int | bool]:
    did_term = f"{treatment_col}:{post_col}"
    pretrend_term = f"{treatment_col}:week_index"

    summary: Dict[str, float | int | bool] = {
        "matched_pairs": int(len(matched_pairs)),
        "did_effect": float(did_model.params[did_term]),
        "did_pvalue": float(did_model.pvalues[did_term]),
        "did_ci_lower": float(did_model.conf_int().loc[did_term, 0]),
        "did_ci_upper": float(did_model.conf_int().loc[did_term, 1]),
        "pretrend_coef": float(pretrend_model.params[pretrend_term]),
        "pretrend_pvalue": float(pretrend_model.pvalues[pretrend_term]),
        "parallel_trends_supported": bool(pretrend_model.pvalues[pretrend_term] > 0.05),
    }
    return summary


def generate_business_summary(summary: Dict[str, float | int | bool]) -> str:
    return (
        f"Matched {summary['matched_pairs']} treated-control user pairs. "
        f"Estimated incremental lift was {summary['did_effect']:.2f} weekly trips "
        f"(95% CI: {summary['did_ci_lower']:.2f} to {summary['did_ci_upper']:.2f}, "
        f"p-value={summary['did_pvalue']:.4f}). "
        + (
            "Pre-treatment trends were not significantly different, supporting the parallel trends assumption."
            if summary["parallel_trends_supported"]
            else "Pre-treatment trends differed significantly, so results should be interpreted with caution."
        )
    )

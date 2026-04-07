from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .config import CampaignConfig
from .diagnostics import balance_table
from .did import build_panel_data, run_did, run_pretrend_check
from .features import build_user_features
from .matching import PropensityMatcher
from .reporting import build_summary, generate_business_summary
from .utils.logging_utils import get_logger


class CampaignEvaluationPipeline:
    def __init__(self, config: CampaignConfig) -> None:
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

    def run(self, users_df: pd.DataFrame, trips_df: pd.DataFrame, treatment_df: pd.DataFrame, weekly_df: pd.DataFrame) -> dict[str, Any]:
        self.logger.info("Building user-level pre-period features")
        feature_df = build_user_features(
            trips_df=trips_df,
            users_df=users_df,
            treatment_df=treatment_df,
            pre_period_end=self.config.pre_period_end,
            user_id_col=self.config.user_id_col,
        )

        feature_cols = [
            "newcomer_density_score",
            "distance_to_station_km",
            "age_band_score",
            "total_trips_pre",
            "active_weeks_pre",
            "recency_days_pre",
            "unique_stations_pre",
            "avg_trips_per_active_week",
        ]

        self.logger.info("Scoring propensity model and matching treated/control users")
        matcher = PropensityMatcher(
            treatment_col=self.config.treatment_col,
            user_id_col=self.config.user_id_col,
            caliper=self.config.caliper,
            random_state=self.config.random_state,
        )
        scored = matcher.fit_score(feature_df, feature_cols)
        matching_result = matcher.match(scored, feature_cols)

        self.logger.info("Computing covariate balance diagnostics")
        balance = balance_table(
            scored_df=matching_result.scored_df,
            matched_pairs=matching_result.matched_pairs,
            feature_cols=matching_result.feature_cols,
            user_id_col=self.config.user_id_col,
        )

        self.logger.info("Building panel dataset and running Difference-in-Differences")
        panel = build_panel_data(
            weekly_df=weekly_df,
            matched_pairs=matching_result.matched_pairs,
            post_period_start=self.config.post_period_start,
            user_id_col=self.config.user_id_col,
            date_col=self.config.date_col,
            post_col=self.config.post_col,
        )
        did_model = run_did(
            panel_df=panel,
            outcome_col=self.config.outcome_col,
            treatment_col=self.config.treatment_col,
            post_col=self.config.post_col,
            user_id_col=self.config.user_id_col,
        )
        pretrend_model = run_pretrend_check(
            panel_df=panel,
            outcome_col=self.config.outcome_col,
            treatment_col=self.config.treatment_col,
            user_id_col=self.config.user_id_col,
            date_col=self.config.date_col,
            post_col=self.config.post_col,
        )

        summary = build_summary(
            did_model=did_model,
            pretrend_model=pretrend_model,
            matched_pairs=matching_result.matched_pairs,
            treatment_col=self.config.treatment_col,
            post_col=self.config.post_col,
        )
        summary["all_features_balanced"] = bool(balance["balanced_flag"].all())
        summary["max_abs_smd"] = float(balance["smd"].abs().max())
        summary["business_readout"] = generate_business_summary(summary)
        return {
            "summary": summary,
            "balance": balance,
            "matched_pairs": matching_result.matched_pairs,
            "panel": panel,
            "config": asdict(self.config),
        }

    def run_from_csv(self, data_dir: str | Path) -> dict[str, Any]:
        data_path = Path(data_dir)
        users = pd.read_csv(data_path / "users.csv")
        trips = pd.read_csv(data_path / "trips.csv")
        treatment = pd.read_csv(data_path / "treatment.csv")
        weekly = pd.read_csv(data_path / "weekly_panel.csv")
        return self.run(users, trips, treatment, weekly)

    def save_artifacts(self, results: dict[str, Any], output_dir: str | Path | None = None) -> None:
        out = Path(output_dir or self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        results["balance"].to_csv(out / "balance_table.csv", index=False)
        results["matched_pairs"].to_csv(out / "matched_pairs.csv", index=False)
        results["panel"].to_csv(out / "matched_panel.csv", index=False)

        with open(out / "summary.json", "w", encoding="utf-8") as f:
            json.dump(results["summary"], f, indent=2)

        with open(out / "business_readout.txt", "w", encoding="utf-8") as f:
            f.write(results["summary"]["business_readout"])

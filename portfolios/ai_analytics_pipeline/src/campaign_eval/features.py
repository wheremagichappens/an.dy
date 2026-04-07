from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def build_user_features(
    trips_df: pd.DataFrame,
    users_df: pd.DataFrame,
    treatment_df: pd.DataFrame,
    pre_period_end: str,
    user_id_col: str = "user_id",
) -> pd.DataFrame:
    trips = trips_df.copy()
    trips["trip_date"] = pd.to_datetime(trips["trip_date"])
    pre_end = pd.to_datetime(pre_period_end)
    hist = trips.loc[trips["trip_date"] <= pre_end].copy()

    if hist.empty:
        raise ValueError("No pre-period trips available to build user features.")

    agg = (
        hist.groupby(user_id_col)
        .agg(
            total_trips_pre=("trip_date", "count"),
            active_weeks_pre=("trip_date", lambda x: x.dt.to_period("W").nunique()),
            last_trip_date=("trip_date", "max"),
            unique_stations_pre=("station_id", "nunique"),
        )
        .reset_index()
    )
    agg["recency_days_pre"] = (pre_end - agg["last_trip_date"]).dt.days
    agg["avg_trips_per_active_week"] = (
        agg["total_trips_pre"] / agg["active_weeks_pre"].replace(0, np.nan)
    ).fillna(0)
    agg = agg.drop(columns=["last_trip_date"])

    df = users_df.merge(agg, on=user_id_col, how="left").merge(treatment_df, on=user_id_col, how="inner")

    numeric_cols: Iterable[str] = [
        "newcomer_density_score",
        "distance_to_station_km",
        "age_band_score",
        "total_trips_pre",
        "active_weeks_pre",
        "recency_days_pre",
        "unique_stations_pre",
        "avg_trips_per_active_week",
    ]
    for col in numeric_cols:
        if col in df.columns:
            fill_value = float(df[col].median()) if not df[col].dropna().empty else 0.0
            df[col] = df[col].fillna(fill_value)

    return df

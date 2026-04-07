from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def generate_mock_data(
    n_users: int = 1500,
    random_state: int = 42,
    start_date: str = "2025-06-02",
    end_date: str = "2025-10-27",
) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    dates = pd.date_range(start_date, end_date, freq="W-MON")
    pre_end = pd.Timestamp("2025-09-01")
    post_start = pd.Timestamp("2025-09-08")

    user_ids = np.arange(1, n_users + 1)
    newcomer_density = rng.beta(2.0, 4.0, size=n_users)
    distance_km = np.clip(rng.gamma(shape=2.0, scale=2.0, size=n_users), 0.1, 20)
    age_band_score = rng.integers(1, 6, size=n_users)
    station_cluster = rng.choice(["urban", "suburban", "regional"], size=n_users, p=[0.45, 0.4, 0.15])

    users = pd.DataFrame(
        {
            "user_id": user_ids,
            "newcomer_density_score": newcomer_density.round(4),
            "distance_to_station_km": distance_km.round(2),
            "age_band_score": age_band_score,
            "station_cluster": station_cluster,
        }
    )

    logit = 1.4 * newcomer_density - 0.18 * distance_km + 0.1 * (age_band_score >= 3)
    p_treat = 1 / (1 + np.exp(-logit))
    treated = rng.binomial(1, np.clip(p_treat, 0.05, 0.95))
    treatment = pd.DataFrame({"user_id": user_ids, "treated": treated})

    base_user_rate = np.clip(
        0.5 + 2.2 * (1 - newcomer_density) + 0.18 * age_band_score - 0.09 * distance_km,
        0.1,
        4.5,
    )

    weekly_rows = []
    trip_rows = []
    station_ids = [f"S{i:03d}" for i in range(1, 26)]

    for idx, user_id in enumerate(user_ids):
        user_treated = treated[idx]
        base_rate = base_user_rate[idx]
        cluster = station_cluster[idx]

        cluster_mult = {"urban": 1.15, "suburban": 0.95, "regional": 0.8}[cluster]

        for dt in dates:
            seasonality = 0.12 * np.sin((dt.weekofyear / 52) * 2 * np.pi)
            post = int(dt >= post_start)
            treatment_effect = 0.0
            if user_treated and post:
                treatment_effect = 0.45 + 0.35 * newcomer_density[idx]

            lam = max(0.05, base_rate * cluster_mult + seasonality + treatment_effect)
            trips = int(rng.poisson(lam))
            weekly_rows.append(
                {
                    "user_id": user_id,
                    "week_start": dt,
                    "weekly_trips": trips,
                    "treated": user_treated,
                }
            )

            if dt <= pre_end:
                for _ in range(trips):
                    trip_day = dt + pd.Timedelta(days=int(rng.integers(0, 7)))
                    station = rng.choice(station_ids)
                    trip_rows.append(
                        {
                            "user_id": user_id,
                            "trip_date": trip_day,
                            "station_id": station,
                        }
                    )

    weekly_panel = pd.DataFrame(weekly_rows)
    trips = pd.DataFrame(trip_rows)
    return {
        "users": users,
        "treatment": treatment,
        "weekly_panel": weekly_panel,
        "trips": trips,
    }


def save_mock_data(output_dir: str | Path, n_users: int = 1500, random_state: int = 42) -> Dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = generate_mock_data(n_users=n_users, random_state=random_state)

    paths: Dict[str, Path] = {}
    for name, df in data.items():
        path = output / f"{name}.csv"
        df.to_csv(path, index=False)
        paths[name] = path
    return paths

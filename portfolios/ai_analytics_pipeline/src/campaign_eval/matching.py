from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class MatchingResult:
    scored_df: pd.DataFrame
    matched_pairs: pd.DataFrame
    feature_cols: List[str]


class PropensityMatcher:
    def __init__(self, treatment_col: str = "treated", user_id_col: str = "user_id", caliper: float = 0.05, random_state: int = 42) -> None:
        self.treatment_col = treatment_col
        self.user_id_col = user_id_col
        self.caliper = caliper
        self.random_state = random_state
        self.model: Pipeline | None = None

    def fit_score(self, user_features: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        df = user_features.copy()
        X = df[feature_cols]
        y = df[self.treatment_col]
        self.model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=self.random_state)),
            ]
        )
        self.model.fit(X, y)
        df["propensity_score"] = self.model.predict_proba(X)[:, 1]
        return df

    def match(self, scored_df: pd.DataFrame, feature_cols: List[str]) -> MatchingResult:
        df = scored_df.copy()
        treated = df[df[self.treatment_col] == 1].copy().reset_index(drop=True)
        control = df[df[self.treatment_col] == 0].copy().reset_index(drop=True)
        if treated.empty or control.empty:
            raise ValueError("Both treated and control populations are required for matching.")

        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(control[["propensity_score"]])
        distances, indices = nn.kneighbors(treated[["propensity_score"]])

        rows = []
        used_controls: set[int] = set()
        for i, (dist, idx) in enumerate(zip(distances.flatten(), indices.flatten())):
            if float(dist) <= self.caliper and int(idx) not in used_controls:
                used_controls.add(int(idx))
                rows.append(
                    {
                        "treated_user_id": int(treated.loc[i, self.user_id_col]),
                        "control_user_id": int(control.loc[int(idx), self.user_id_col]),
                        "treated_ps": float(treated.loc[i, "propensity_score"]),
                        "control_ps": float(control.loc[int(idx), "propensity_score"]),
                        "distance": float(dist),
                    }
                )

        matched_pairs = pd.DataFrame(rows)
        if matched_pairs.empty:
            raise ValueError("No matches found within the specified caliper.")

        return MatchingResult(scored_df=df, matched_pairs=matched_pairs, feature_cols=feature_cols)

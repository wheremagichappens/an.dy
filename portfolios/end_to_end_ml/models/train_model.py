from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

DATA_PATH = Path("data/processed/customer_features_labeled.csv")
MODEL_DIR = Path("artifacts/models")
METRICS_DIR = Path("artifacts/metrics")

TARGET_COL = "churned"
DROP_COLS = ["first_purchase_date", "last_purchase_date", "future_purchase_count"]
RANDOM_SEED = 42

@dataclass
class TrainResult:
    name: str
    pipeline: Pipeline
    metrics: Dict[str, float]

def _load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def _prep_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL])

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]
    return X, y

def _build_preprocess(numeric_cols) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("num", numeric_pipe, numeric_cols)],
        remainder="drop",
    )

def _evaluate(model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
    proba = model.predict_proba(X_val)[:, 1]
    pred = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_val, proba)
    acc = accuracy_score(y_val, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_val, pred, average="binary", zero_division=0
    )

    return {
        "roc_auc": float(auc),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }

def train_logistic(X_train, y_train, X_val, y_val) -> TrainResult:
    preprocess = _build_preprocess(X_train.columns.tolist())
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    pipe = Pipeline(steps=[("prep", preprocess), ("clf", clf)])
    pipe.fit(X_train, y_train)
    metrics = _evaluate(pipe, X_val, y_val)
    return TrainResult(name="logreg", pipeline=pipe, metrics=metrics)

def train_gbm(X_train, y_train, X_val, y_val) -> TrainResult:
    numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    preprocess = ColumnTransformer(
        transformers=[("num", numeric_pipe, X_train.columns.tolist())],
        remainder="drop",
    )
    clf = GradientBoostingClassifier(random_state=RANDOM_SEED)
    pipe = Pipeline(steps=[("prep", preprocess), ("clf", clf)])
    pipe.fit(X_train, y_train)
    metrics = _evaluate(pipe, X_val, y_val)
    return TrainResult(name="gbm", pipeline=pipe, metrics=metrics)

def _save_artifacts(result: TrainResult) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / f"{result.name}.joblib"
    metrics_path = METRICS_DIR / f"{result.name}.json"

    joblib.dump(result.pipeline, model_path)
    with open(metrics_path, "w") as f:
        json.dump(result.metrics, f, indent=2)

    print(f"[saved] model:   {model_path}")
    print(f"[saved] metrics: {metrics_path}")

def main() -> None:
    df = _load_data(DATA_PATH)
    X, y = _prep_xy(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    results = [train_logistic(X_train, y_train, X_val, y_val),
               train_gbm(X_train, y_train, X_val, y_val)]

    print("\n=== Model comparison ===")
    for r in results:
        print(
            f"{r.name:>6} | AUC={r.metrics['roc_auc']:.4f} | F1={r.metrics['f1']:.4f} "
            f"| P={r.metrics['precision']:.4f} R={r.metrics['recall']:.4f} Acc={r.metrics['accuracy']:.4f}"
        )

    for r in results:
        _save_artifacts(r)

if __name__ == "__main__":
    main()

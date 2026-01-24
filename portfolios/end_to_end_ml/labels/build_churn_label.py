import pandas as pd
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
RAW_DATA_PATH = Path("data/raw/online_retail.csv")
FEATURES_PATH = Path("data/processed/customer_features.csv")
OUTPUT_PATH = Path("data/processed/customer_features_labeled.csv")

SNAPSHOT_DATE = pd.to_datetime("2011-12-01")
CHURN_WINDOW_DAYS = 30

def load_transactions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="ISO-8859-1")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df

def load_features(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def build_churn_label(transactions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    churn_window_end = SNAPSHOT_DATE + pd.Timedelta(days=CHURN_WINDOW_DAYS)

    future_purchases = transactions[
        (transactions["InvoiceDate"] >= SNAPSHOT_DATE)
        & (transactions["InvoiceDate"] < churn_window_end)
        & (transactions["CustomerID"].notna())
    ]

    future_activity = (
        future_purchases.groupby("CustomerID")
        .size()
        .reset_index(name="future_purchase_count")
    )

    labeled_df = features.merge(future_activity, on="CustomerID", how="left")
    labeled_df["future_purchase_count"] = labeled_df["future_purchase_count"].fillna(0)

    # 1 = churned, 0 = retained
    labeled_df["churned"] = (labeled_df["future_purchase_count"] == 0).astype(int)
    return labeled_df

def main():
    transactions = load_transactions(RAW_DATA_PATH)
    features = load_features(FEATURES_PATH)

    labeled_df = build_churn_label(transactions, features)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    labeled_df.to_csv(OUTPUT_PATH, index=False)

    churn_rate = labeled_df["churned"].mean()
    print(f"Saved labeled dataset to {OUTPUT_PATH}")
    print(f"Churn rate: {churn_rate:.2%}")
    print(f"Shape: {labeled_df.shape}")

if __name__ == "__main__":
    main()

import pandas as pd
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
RAW_DATA_PATH = Path("data/raw/online_retail.csv")
OUTPUT_PATH = Path("data/processed/customer_features.csv")
SNAPSHOT_DATE = pd.to_datetime("2011-12-01")

# -----------------------------
# Load data
# -----------------------------
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="ISO-8859-1")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df

# -----------------------------
# Build customer-level features
# -----------------------------
def build_customer_features(df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    df = df.dropna(subset=["CustomerID"]).copy()
    df = df[df["InvoiceDate"] < snapshot_date].copy()

    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    customer_df = (
        df.groupby("CustomerID")
        .agg(
            last_purchase_date=("InvoiceDate", "max"),
            first_purchase_date=("InvoiceDate", "min"),
            purchase_count=("InvoiceNo", "nunique"),
            total_revenue=("TotalPrice", "sum"),
            avg_order_value=("TotalPrice", "mean"),
            unique_products=("StockCode", "nunique"),
        )
        .reset_index()
    )

    # Time-based features
    customer_df["recency_days"] = (snapshot_date - customer_df["last_purchase_date"]).dt.days
    customer_df["customer_lifetime_days"] = (
        customer_df["last_purchase_date"] - customer_df["first_purchase_date"]
    ).dt.days.clip(lower=1)

    customer_df["purchase_frequency"] = (
        customer_df["purchase_count"] / customer_df["customer_lifetime_days"]
    )

    return customer_df

def main():
    df = load_data(RAW_DATA_PATH)
    customer_features = build_customer_features(df, SNAPSHOT_DATE)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    customer_features.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved customer features to {OUTPUT_PATH}")
    print(f"Shape: {customer_features.shape}")

if __name__ == "__main__":
    main()

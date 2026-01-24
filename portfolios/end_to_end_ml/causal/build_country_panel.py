from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/online_retail.csv")
OUT_PATH = Path("data/processed/country_week_panel.csv")

def main():
    df = pd.read_csv(RAW_PATH, encoding="ISO-8859-1")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate", "Country"])
    df["revenue"] = df["Quantity"] * df["UnitPrice"]
    df["week"] = df["InvoiceDate"].dt.to_period("W").dt.start_time

    panel = (
        df.groupby(["Country", "week"])
        .agg(
            trips=("InvoiceNo", "nunique"),
            buyers=("CustomerID", "nunique"),
            revenue=("revenue", "sum"),
        )
        .reset_index()
        .sort_values(["Country", "week"])
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT_PATH, index=False)
    print(f"Saved panel to {OUT_PATH} | shape={panel.shape}")

if __name__ == "__main__":
    main()

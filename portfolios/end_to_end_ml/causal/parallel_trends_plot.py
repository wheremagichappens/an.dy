from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PANEL_PATH = Path("data/processed/country_week_panel.csv")
OUT_PATH = Path("artifacts/metrics/parallel_trends_plot.png")

POLICY_DATE = "2011-09-01"
TREATED_COUNTRIES = {"United Kingdom"}
OUTCOME = "revenue"  # or "trips" or "buyers"

def main():
    df = pd.read_csv(PANEL_PATH)
    df["week"] = pd.to_datetime(df["week"], errors="coerce")
    df = df.dropna(subset=["week"])

    df["group"] = df["Country"].apply(lambda c: "treated" if c in TREATED_COUNTRIES else "control")
    agg = df.groupby(["group", "week"])[OUTCOME].mean().reset_index()

    treated = agg[agg["group"] == "treated"]
    control = agg[agg["group"] == "control"]

    plt.figure(figsize=(10, 5))
    plt.plot(treated["week"], treated[OUTCOME], label="treated (avg)")
    plt.plot(control["week"], control[OUTCOME], label="control (avg)")
    plt.axvline(pd.to_datetime(POLICY_DATE), linestyle="--", label="policy date")
    plt.title(f"Parallel Trends Check: {OUTCOME}")
    plt.xlabel("Week")
    plt.ylabel(OUTCOME)
    plt.legend()
    plt.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH)
    print(f"Saved plot to {OUT_PATH}")

if __name__ == "__main__":
    main()

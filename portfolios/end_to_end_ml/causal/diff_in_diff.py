from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf

PANEL_PATH = Path("data/processed/country_week_panel.csv")
OUT_PATH = Path("artifacts/metrics/did_results.csv")

POLICY_DATE = "2011-09-01"
TREATED_COUNTRIES = {"United Kingdom"}
OUTCOME = "revenue"  # or "trips" or "buyers"

def main():
    df = pd.read_csv(PANEL_PATH)
    df["week"] = pd.to_datetime(df["week"], errors="coerce")
    df = df.dropna(subset=["week"])

    df["treated"] = df["Country"].isin(TREATED_COUNTRIES).astype(int)
    df["post"] = (df["week"] >= pd.to_datetime(POLICY_DATE)).astype(int)
    df["did"] = df["treated"] * df["post"]

    # Two-way fixed effects (country + week) with clustered SE by country
    model = smf.ols(
        f"{OUTCOME} ~ treated + post + did + C(Country) + C(week)",
        data=df
    ).fit(cov_type="cluster", cov_kwds={"groups": df["Country"]})

    out = pd.DataFrame([{
        "outcome": OUTCOME,
        "policy_date": POLICY_DATE,
        "treated_countries": ",".join(sorted(TREATED_COUNTRIES)),
        "did_effect": float(model.params.get("did", float("nan"))),
        "p_value": float(model.pvalues.get("did", float("nan"))),
        "n_rows": int(len(df)),
    }])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print("=== DiD Results ===")
    print(out.to_string(index=False))
    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()

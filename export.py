"""Export portfolio-level tables for a BI dashboard (Power BI, Tableau, Looker…).

Turns the scored held-out portfolio into clean, aggregated CSVs a BI tool can connect to.
Run after train.py. Outputs land in dashboard/.

Run with:  python export.py
"""

import json
import os

import joblib
import pandas as pd

import loan_default as L
import fairness as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "dashboard")


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = json.load(open(os.path.join(HERE, "model_meta.json"), encoding="utf-8"))
    t = meta["threshold"]

    scored = F.load_scored()
    scored["decision"] = pd.cut(scored["proba"], [-0.01, max(0, t - 0.15), t, 1.01],
                                labels=["approve", "review", "decline"])
    scored["risk_band"] = pd.cut(scored["proba"], [-0.01, 0.1, 0.25, 0.5, 1.01],
                                 labels=["very low", "low", "medium", "high"])
    scored.round(4).to_csv(os.path.join(OUT, "scored_portfolio.csv"), index=False)

    by_prov = (scored.groupby("province", observed=True)
               .agg(n=("actual", "size"), default_rate=("actual", "mean"),
                    avg_risk=("proba", "mean"),
                    approval_rate=("decision", lambda s: (s == "approve").mean()))
               .round(4).reset_index())
    by_prov.to_csv(os.path.join(OUT, "risk_by_province.csv"), index=False)

    pd.DataFrame([meta["metrics"]]).to_csv(os.path.join(OUT, "model_metrics.csv"), index=False)

    rf = joblib.load(os.path.join(HERE, "rf_base.joblib"))
    imp = (pd.Series(rf.named_steps["clf"].feature_importances_, index=L.feature_names(rf))
           .sort_values(ascending=False).round(4).rename("importance").reset_index()
           .rename(columns={"index": "feature"}))
    imp.to_csv(os.path.join(OUT, "feature_importance.csv"), index=False)

    pd.read_csv(os.path.join(HERE, "data", "boc_macro.csv")).to_csv(
        os.path.join(OUT, "macro_environment.csv"), index=False)

    print("Exported to dashboard/:")
    for f in ("scored_portfolio.csv", "risk_by_province.csv", "model_metrics.csv",
              "feature_importance.csv", "macro_environment.csv"):
        print(f"  {f}")
    print(f"\nPortfolio: {len(scored):,} loans · "
          f"{(scored['decision'] == 'approve').mean():.0%} approved · "
          f"{scored['actual'].mean():.0%} default rate")


if __name__ == "__main__":
    main()

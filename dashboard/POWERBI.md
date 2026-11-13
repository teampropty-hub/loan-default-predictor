# BI Dashboard (Power BI / Tableau)

The model's portfolio-level results are exported here as clean CSVs so they can be wired
straight into a BI tool. Regenerate them any time with `python export.py`.

## Files

| File | Grain | Use |
|---|---|---|
| `scored_portfolio.csv` | one row per scored loan | detail table; slice by province, decision, risk band |
| `risk_by_province.csv` | one row per province | map / bar: default rate, approval rate, avg risk |
| `model_metrics.csv` | one row | KPI cards (AUC, recall, Brier, ECE) |
| `feature_importance.csv` | one row per feature | bar: what drives risk |
| `macro_environment.csv` | one row per month | line: Bank of Canada policy rate over time |

## Build it in Power BI Desktop

1. **Get Data → Text/CSV** and load the five files from this folder.
2. Suggested pages:
   - **Overview** — KPI cards from `model_metrics.csv`; a decision-mix donut from
     `scored_portfolio.csv` (`decision`).
   - **Regional risk** — a filled map of Canada keyed on `province` from
     `risk_by_province.csv`, coloured by `default_rate`.
   - **Drivers** — a bar chart of `feature_importance.csv`.
   - **Macro** — a line of `boc_policy_rate` from `macro_environment.csv`.
3. Save as `Canadian_Credit_Risk_Dashboard.pbix`.

> A live, no-install version of the same views ships in the app's **Portfolio** tab
> (`streamlit run app.py`).

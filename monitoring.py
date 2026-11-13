"""Model monitoring — data-drift detection with the Population Stability Index (PSI).

OSFI Guideline E-23 expects models to be monitored over their lifecycle, not just built
once. A credit model silently degrades when the population it scores drifts away from the
one it was trained on (rate shocks, a new segment, a macro shift). PSI is the standard
early-warning metric: PSI < 0.10 ≈ stable, 0.10–0.25 ≈ moderate shift, > 0.25 ≈ significant
drift that warrants revalidation.

Run with:  python monitoring.py
"""

import numpy as np

import loan_default as L


def population_stability_index(reference, current, bins=10):
    """PSI of ``current`` vs ``reference`` for one numeric feature."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    ref_pct = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_pct = np.histogram(cur, bins=edges)[0] / len(cur)
    ref_pct = np.clip(ref_pct, 1e-4, None)
    cur_pct = np.clip(cur_pct, 1e-4, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_report(reference_df, current_df, features):
    rows = []
    for f in features:
        psi = population_stability_index(reference_df[f], current_df[f])
        flag = "OK" if psi < 0.10 else "WATCH" if psi < 0.25 else "DRIFT"
        rows.append((f, round(psi, 3), flag))
    return rows


def _print(title, rows):
    print(f"\n{title}")
    print(f"  {'feature':<22}{'PSI':>8}  status")
    for f, psi, flag in rows:
        print(f"  {f:<22}{psi:>8.3f}  {flag}")


def main():
    df = L.engineer_features(L.load_data())
    feats = L.NUMERIC
    half = len(df) // 2
    reference, current = df.iloc[:half], df.iloc[half:]

    _print("Reference vs a fresh same-period batch (expect no drift):",
           drift_report(reference, current, feats))

    # Simulate a macro shock: incomes fall 15% and rates rise 3 points in the new batch.
    shocked = current.copy()
    shocked["person_income"] *= 0.85
    shocked["loan_int_rate"] = shocked["loan_int_rate"] + 3
    shocked = L.engineer_features(shocked.drop(columns=L.ENGINEERED))
    _print("Reference vs a simulated rate/income shock (expect drift):",
           drift_report(reference, shocked, feats))

    print("\nPSI < 0.10 stable | 0.10-0.25 moderate | > 0.25 significant drift -> revalidate.")


if __name__ == "__main__":
    main()

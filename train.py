"""Train the production model and save everything the app and reports need.

Outputs (all consumed by app.py / fairness.py):
  * model.joblib        — Random Forest refit on ALL data, compressed
  * model_meta.json     — cost-optimal threshold, metrics, medians, cost params
  * data/test_scored.csv— held-out rows with probabilities (for live threshold + fairness)
  * reports/*.png       — ROC, confusion, feature importance, cost-vs-threshold

Run with:  python train.py
"""

import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import loan_default as L
import threshold as TH

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "model.joblib")
META = os.path.join(HERE, "model_meta.json")
TEST_SCORED = os.path.join(HERE, "data", "test_scored.csv")


def main():
    os.makedirs(L.REPORTS, exist_ok=True)
    df = L.engineer_features(L.load_data())
    X, y = df[L.RAW_CATEGORICAL + L.NUMERIC], df[L.TARGET]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = L.build_models()["Random Forest"].fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy": round(accuracy_score(y_te, pred), 4),
        "precision": round(precision_score(y_te, pred), 4),
        "recall": round(recall_score(y_te, pred), 4),
        "f1": round(f1_score(y_te, pred), 4),
        "roc_auc": round(roc_auc_score(y_te, proba), 4),
    }

    # Cost-optimal threshold on the held-out set, plus the charts.
    best = TH.plot_cost_curve(y_te, proba, X_te["loan_amnt"], X_te["loan_int_rate"])
    L._save_roc({"Random Forest": model}, X_te, y_te)
    L._save_confusion(model, X_te, y_te)
    L._save_importances(model)

    # Save the scored test rows so the app can recompute the threshold and run
    # fairness without retraining.
    scored = pd.DataFrame({
        "proba": proba,
        "actual": y_te.values,
        "loan_amnt": X_te["loan_amnt"].values,
        "loan_int_rate": X_te["loan_int_rate"].values,
        "person_age": X_te["person_age"].values,
        "person_home_ownership": X_te["person_home_ownership"].values,
    })
    scored.to_csv(TEST_SCORED, index=False)

    # Refit on ALL data for the artifact we actually ship. A lighter forest (120 trees)
    # keeps the file small with effectively identical predictions.
    production = Pipeline([
        ("prep", L.build_preprocessor()),
        ("clf", RandomForestClassifier(n_estimators=120, random_state=42,
                                       class_weight="balanced", n_jobs=-1)),
    ]).fit(X, y)
    joblib.dump(production, MODEL, compress=3)

    meta = {
        "model": "RandomForestClassifier",
        "threshold": round(best["threshold"], 3),
        "cost_params": {"lgd": TH.DEFAULT_LGD, "term_years": TH.DEFAULT_TERM},
        "cost_saved_vs_0.5": round(best["saved"], 2),
        "metrics": metrics,
        "medians": {c: float(df[c].median()) for c in L.NUMERIC},
        "categorical": L.RAW_CATEGORICAL,
        "numeric": L.NUMERIC,
    }
    with open(META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    size_mb = os.path.getsize(MODEL) / 1e6
    print(f"metrics: {metrics}")
    print(f"optimal threshold: {meta['threshold']}  "
          f"(saves ${best['saved']:,.0f} vs 0.50 on the test set)")
    print(f"saved model.joblib ({size_mb:.1f} MB), model_meta.json, data/test_scored.csv")


if __name__ == "__main__":
    main()

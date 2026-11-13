"""Train the production model and save everything the app and reports need.

Outputs:
  * model.joblib        — isotonic-calibrated Random Forest, refit on ALL data (probabilities)
  * rf_base.joblib      — plain Random Forest, refit on ALL data (SHAP explanations)
  * model_meta.json     — threshold, metrics (incl. Brier/ECE), medians, cost params
  * data/test_scored.csv— held-out rows with calibrated probabilities (threshold + fairness)
  * reports/*.png       — ROC, confusion, importance, cost, calibration, SHAP summary

Run with:  python train.py   (run data/canadianize.py first)
"""

import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import loan_default as L
import threshold as TH
import calibration as C
import explain as E

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "model.joblib")
RF_BASE = os.path.join(HERE, "rf_base.joblib")
META = os.path.join(HERE, "model_meta.json")
TEST_SCORED = os.path.join(HERE, "data", "test_scored.csv")


def _light_rf():
    return Pipeline([
        ("prep", L.build_preprocessor()),
        ("clf", RandomForestClassifier(n_estimators=120, random_state=42,
                                       class_weight="balanced", n_jobs=-1)),
    ])


def main():
    os.makedirs(L.REPORTS, exist_ok=True)
    df = L.engineer_features(L.load_data())
    X, y = df[L.RAW_CATEGORICAL + L.NUMERIC], df[L.TARGET]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Plain RF for ranking charts / SHAP; calibrated model for trustworthy probabilities.
    rf = L.build_models()["Random Forest"].fit(X_tr, y_tr)
    cal = C.build_calibrated().fit(X_tr, y_tr)

    raw_proba = rf.predict_proba(X_te)[:, 1]
    proba = cal.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "accuracy": round(accuracy_score(y_te, pred), 4),
        "precision": round(precision_score(y_te, pred), 4),
        "recall": round(recall_score(y_te, pred), 4),
        "f1": round(f1_score(y_te, pred), 4),
        "roc_auc": round(roc_auc_score(y_te, proba), 4),
        "brier": round(brier_score_loss(y_te, proba), 4),
        "ece": round(C.expected_calibration_error(y_te, proba), 4),
    }

    # Cost-optimal threshold on calibrated probabilities + all the charts.
    best = TH.plot_cost_curve(y_te, proba, X_te["loan_amnt"], X_te["loan_int_rate"])
    L._save_roc({"Random Forest": rf}, X_te, y_te)
    L._save_confusion(rf, X_te, y_te)
    L._save_importances(rf)
    C._reliability_plot(y_te, raw_proba, proba, os.path.join(L.REPORTS, "calibration_curve.png"))
    E.save_global_summary(rf, X_tr)

    scored = pd.DataFrame({
        "proba": proba, "actual": y_te.values,
        "loan_amnt": X_te["loan_amnt"].values, "loan_int_rate": X_te["loan_int_rate"].values,
        "person_age": X_te["person_age"].values, "province": X_te["province"].values,
        "credit_score": X_te["credit_score"].values,
    })
    scored.to_csv(TEST_SCORED, index=False)

    # Refit on ALL data for the shipped artifacts.
    joblib.dump(C.build_calibrated(_light_rf()).fit(X, y), MODEL, compress=3)
    joblib.dump(_light_rf().fit(X, y), RF_BASE, compress=3)

    meta = {
        "model": "Isotonic-calibrated RandomForestClassifier",
        "currency": "CAD",
        "threshold": round(best["threshold"], 3),
        "cost_params": {"lgd": TH.DEFAULT_LGD, "term_years": TH.DEFAULT_TERM},
        "cost_saved_vs_0.5": round(best["saved"], 2),
        "metrics": metrics,
        "medians": {c: float(df[c].median()) for c in L.NUMERIC},
        "categorical": L.RAW_CATEGORICAL, "numeric": L.NUMERIC,
    }
    with open(META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"metrics: {metrics}")
    print(f"optimal threshold: {meta['threshold']} (saves ${best['saved']:,.0f} CAD vs 0.50)")
    print(f"saved model.joblib ({os.path.getsize(MODEL)/1e6:.1f} MB), rf_base.joblib, "
          f"model_meta.json, data/test_scored.csv")


if __name__ == "__main__":
    main()

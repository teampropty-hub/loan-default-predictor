"""Explainability with SHAP — why did the model score this applicant?

OSFI Guideline E-23 expects models to be explainable, and lenders must give a declined
applicant the reasons (adverse-action notice). SHAP attributes each prediction to the
features that pushed it up or down, both globally (which features matter overall) and for a
single applicant (the reasons for their score).

SHAP explains the underlying Random Forest (`rf_base.joblib`), which is the ranking engine;
the calibrated model only rescales its probabilities.
"""

import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
import shap

import loan_default as L

REPORTS = L.REPORTS


def _rf_parts(rf_pipeline):
    """Return (preprocessor, classifier, feature_names) from a plain RF pipeline."""
    prep = rf_pipeline.named_steps["prep"]
    clf = rf_pipeline.named_steps["clf"]
    return prep, clf, L.feature_names(rf_pipeline)


def _to_dense(mat):
    return mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)


def shap_values_for(rf_pipeline, X):
    """SHAP values (positive class) for the rows in X, plus the feature names."""
    prep, clf, names = _rf_parts(rf_pipeline)
    Xt = _to_dense(prep.transform(X))
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):           # [class0, class1]
        sv = sv[1]
    elif getattr(sv, "ndim", 2) == 3:  # (n, features, classes)
        sv = sv[:, :, 1]
    return sv, Xt, names


def top_reasons(rf_pipeline, applicant_feats, k=4):
    """Top features driving one applicant's score: [(feature, signed_contribution)]."""
    sv, _, names = shap_values_for(rf_pipeline, applicant_feats)
    contribs = sorted(zip(names, sv[0]), key=lambda kv: abs(kv[1]), reverse=True)
    return contribs[:k]


def save_global_summary(rf_pipeline, X, path=None, sample=400):
    import matplotlib.pyplot as plt
    Xs = X.sample(min(sample, len(X)), random_state=42)
    sv, Xt, names = shap_values_for(rf_pipeline, Xs)
    shap.summary_plot(sv, Xt, feature_names=names, show=False, max_display=12)
    path = path or os.path.join(REPORTS, "shap_summary.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout(); plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
    return path


def main():
    from sklearn.model_selection import train_test_split
    df = L.engineer_features(L.load_data())
    X, y = df[L.RAW_CATEGORICAL + L.NUMERIC], df[L.TARGET]
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = L.build_models()["Random Forest"].fit(X_tr, y_tr)
    path = save_global_summary(rf, X_tr)
    print(f"SHAP summary -> {os.path.relpath(path, L.HERE)}")

    # Example adverse-action reasons for the first pending applicant.
    import pandas as pd
    req = L.prepare_features(pd.read_csv(L.REQUESTS), df[L.NUMERIC].median())
    print("\nTop reasons for applicant 1:")
    for feat, val in top_reasons(rf, req.iloc[[0]]):
        direction = "raises risk" if val > 0 else "lowers risk"
        print(f"  {feat:<28} {direction} ({val:+.3f})")


if __name__ == "__main__":
    main()

"""Loan default risk — model training, comparison and applicant scoring.

Pipeline:
  1. load historical loans and engineer a few interpretable features
  2. compare three classifiers with cross-validated ROC AUC
  3. evaluate the best on a held-out slice (accuracy / precision / recall / F1 / AUC)
  4. save charts (ROC curves, confusion matrix, feature importance)
  5. score pending applicants and turn probabilities into approve / review / decline

Run with:  python loan_default.py
The reusable pieces (load_data, engineer_features, build_models...) are imported by
analysis.ipynb so the notebook and the script never drift apart.
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless: write charts to files, never open a window
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
# Canadian-framed data (CAD, province, 300-900 credit score); see data/canadianize.py.
DATA = os.path.join(HERE, "data", "ca_credit_train.csv")
REQUESTS = os.path.join(HERE, "data", "ca_loan_requests.csv")
REPORTS = os.path.join(HERE, "reports")
TARGET = "loan_status"  # 1 = defaulted, 0 = repaid

RAW_CATEGORICAL = ["person_home_ownership", "loan_intent", "loan_grade",
                   "cb_person_default_on_file", "province"]
RAW_NUMERIC = ["person_age", "person_income", "person_emp_length", "loan_amnt",
               "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
               "credit_score"]
ENGINEERED = ["loan_to_income", "credit_history_ratio", "income_per_age"]
NUMERIC = RAW_NUMERIC + ENGINEERED


def load_data(path=DATA):
    """Load the training data and impute the two columns with missing values."""
    df = pd.read_csv(path)
    for col in ("person_emp_length", "loan_int_rate"):
        df[col] = df[col].fillna(df[col].median())
    # Employment length has impossible outliers (>80 years) — cap them.
    df["person_emp_length"] = df["person_emp_length"].clip(upper=47)
    df["person_age"] = df["person_age"].clip(upper=90)
    return df


def engineer_features(df):
    """Add a few interpretable ratios a credit analyst would actually look at."""
    df = df.copy()
    df["loan_to_income"] = df["loan_amnt"] / df["person_income"].clip(lower=1)
    df["credit_history_ratio"] = df["cb_person_cred_hist_length"] / df["person_age"].clip(lower=1)
    df["income_per_age"] = df["person_income"] / df["person_age"].clip(lower=1)
    return df


def build_preprocessor():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), RAW_CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ])


def build_models():
    """The three classifiers we compare, each wrapped with the preprocessor."""
    pre = build_preprocessor
    return {
        "Logistic Regression": Pipeline([("prep", pre()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "Random Forest": Pipeline([("prep", pre()),
            ("clf", RandomForestClassifier(n_estimators=250, random_state=42,
                                           class_weight="balanced", n_jobs=-1))]),
        "Gradient Boosting": Pipeline([("prep", pre()),
            ("clf", GradientBoostingClassifier(random_state=42))]),
    }


def feature_names(fitted_pipeline):
    """Recover readable feature names after one-hot encoding."""
    pre = fitted_pipeline.named_steps["prep"]
    cat = list(pre.named_transformers_["cat"].get_feature_names_out(RAW_CATEGORICAL))
    return cat + NUMERIC


def _save_roc(models_fitted, X_test, y_test):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    for name, model in models_fitted.items():
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC {auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC — default detection"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(os.path.join(REPORTS, "roc_curves.png"), dpi=120)
    plt.close()


def _save_confusion(model, X_test, y_test):
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["repaid", "default"]); ax.set_yticklabels(["repaid", "default"])
    ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title("Confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout(); plt.savefig(os.path.join(REPORTS, "confusion_matrix.png"), dpi=120)
    plt.close()


def _save_importances(model):
    import matplotlib.pyplot as plt
    clf = model.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return
    names = feature_names(model)
    pairs = sorted(zip(names, clf.feature_importances_), key=lambda kv: kv[1])[-12:]
    labels, vals = zip(*pairs)
    plt.figure(figsize=(6.5, 5))
    plt.barh(labels, vals, color="#2563eb")
    plt.title("Top features driving default risk"); plt.xlabel("importance")
    plt.tight_layout(); plt.savefig(os.path.join(REPORTS, "feature_importance.png"), dpi=120)
    plt.close()


def prepare_features(df, train_medians):
    """Engineer + align features and fill any gaps with training medians (no leakage).

    Shared by score_requests, fairness.py and the Streamlit app so every path feeds the
    model exactly the same columns in the same order.
    """
    feats = engineer_features(df).reindex(columns=RAW_CATEGORICAL + NUMERIC)
    for col in NUMERIC:
        feats[col] = feats[col].fillna(train_medians[col])
    return feats


def score_requests(model, train_df, threshold=0.5):
    """Score the pending applicants and attach an approve/review/decline action."""
    medians = train_df[NUMERIC].median()
    requests = pd.read_csv(REQUESTS)
    feats = prepare_features(requests, medians)
    requests["default_risk"] = model.predict_proba(feats)[:, 1]
    # Decline at/above the threshold; a review band sits just below it.
    review_floor = max(0.0, threshold - 0.15)
    requests["action"] = pd.cut(
        requests["default_risk"], [-0.01, review_floor, threshold, 1.01],
        labels=["approve", "review", "DECLINE"])
    return requests.sort_values("default_risk", ascending=False)


def main():
    os.makedirs(REPORTS, exist_ok=True)
    df = engineer_features(load_data())
    X, y = df[RAW_CATEGORICAL + NUMERIC], df[TARGET]
    print(f"Training on {len(X):,} past loans ({y.mean():.1%} defaulted).\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print("5-fold cross-validated ROC AUC (training set):")
    fitted, scores = {}, {}
    for name, model in build_models().items():
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
        model.fit(X_train, y_train)
        fitted[name] = model
        scores[name] = cv.mean()
        print(f"  {name:<20} {cv.mean():.3f}  (+/- {cv.std():.3f})")

    best_name = max(scores, key=scores.get)
    best = fitted[best_name]
    print(f"\nBest model: {best_name}\n")

    print(f"{'model':<20}{'acc':>7}{'prec':>7}{'recall':>8}{'F1':>7}{'AUC':>7}")
    for name, model in fitted.items():
        p = model.predict(X_test); pr = model.predict_proba(X_test)[:, 1]
        print(f"{name:<20}{accuracy_score(y_test, p):>7.1%}"
              f"{precision_score(y_test, p):>7.1%}{recall_score(y_test, p):>8.1%}"
              f"{f1_score(y_test, p):>7.1%}{roc_auc_score(y_test, pr):>7.3f}")

    _save_roc(fitted, X_test, y_test)
    _save_confusion(best, X_test, y_test)
    _save_importances(best)
    print(f"\nCharts written to {os.path.relpath(REPORTS, HERE)}/")

    ranked = score_requests(best, df)
    print("\nPending loan requests, riskiest first:")
    for _, r in ranked.iterrows():
        print(f"  {r.get('borrower', 'applicant'):<20} {r['default_risk']:5.1%}  -> {r['action']}")


if __name__ == "__main__":
    main()

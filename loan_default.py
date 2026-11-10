"""Predict loan default risk from borrower and loan features.

Trains a classifier on historical credit-risk data (whether each loan defaulted),
measures it on a held-out slice, then scores a small file of pending loan requests
and ranks the applicants by estimated default risk.

Run with:  python loan_default.py
"""

import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = os.path.dirname(__file__)
TRAIN = os.path.join(HERE, "data", "credit_risk_train.csv")
REQUESTS = os.path.join(HERE, "data", "loan_requests.csv")
TARGET = "loan_status"  # 1 = defaulted, 0 = repaid

CATEGORICAL = ["person_home_ownership", "loan_intent", "loan_grade",
               "cb_person_default_on_file"]
NUMERIC = ["person_age", "person_income", "person_emp_length", "loan_amnt",
           "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]


def build_model():
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])
    return Pipeline([
        ("prep", pre),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42,
                                      class_weight="balanced", n_jobs=-1)),
    ])


def load_training():
    df = pd.read_csv(TRAIN)
    # A couple of numeric columns carry missing values; fill with the median.
    for col in ("person_emp_length", "loan_int_rate"):
        df[col] = df[col].fillna(df[col].median())
    return df[CATEGORICAL + NUMERIC], df[TARGET]


def main():
    X, y = load_training()
    print(f"Training on {len(X):,} past loans ({y.mean():.1%} defaulted).")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    print("\nHeld-out performance:")
    print(f"  accuracy: {accuracy_score(y_test, preds):.1%}")
    print(f"  ROC AUC : {roc_auc_score(y_test, proba):.3f}")
    print(f"  caught {tp} of {tp + fn} real defaults "
          f"({tp / (tp + fn):.0%} recall), {fp} false alarms")

    # Score the pending requests and rank by risk.
    requests = pd.read_csv(REQUESTS)
    feats = requests.reindex(columns=CATEGORICAL + NUMERIC)
    for col in ("person_emp_length", "loan_int_rate"):
        feats[col] = feats[col].fillna(X[col].median())
    requests["default_risk"] = model.predict_proba(feats)[:, 1]

    print("\nPending loan requests, riskiest first:")
    ranked = requests.sort_values("default_risk", ascending=False)
    for _, row in ranked.iterrows():
        flag = "DECLINE" if row["default_risk"] >= 0.5 else "review " if row["default_risk"] >= 0.2 else "approve"
        name = row.get("borrower", "applicant")
        print(f"  {name:<20} {row['default_risk']:5.1%}  -> {flag}")


if __name__ == "__main__":
    main()

"""Hyperparameter tuning for the Random Forest (RandomizedSearchCV).

Searches the main Random Forest knobs by cross-validated ROC AUC on a subsample (for
speed) and prints the best configuration. Run it when you want to re-tune; the winning
parameters can then be folded into `build_models()` in loan_default.py.

Run with:  python tune.py
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

import loan_default as L


def main():
    df = L.engineer_features(L.load_data())
    X, y = df[L.RAW_CATEGORICAL + L.NUMERIC], df[L.TARGET]
    # Subsample for a fast search.
    idx = X.sample(min(8000, len(X)), random_state=42).index
    X, y = X.loc[idx], y.loc[idx]

    pipe = Pipeline([("prep", L.build_preprocessor()),
                     ("clf", RandomForestClassifier(random_state=42, class_weight="balanced",
                                                    n_jobs=-1))])
    grid = {
        "clf__n_estimators": [120, 200, 300],
        "clf__max_depth": [None, 8, 12, 20],
        "clf__min_samples_leaf": [1, 2, 5, 10],
        "clf__max_features": ["sqrt", "log2", 0.5],
    }
    search = RandomizedSearchCV(pipe, grid, n_iter=12, scoring="roc_auc", cv=3,
                                random_state=42, n_jobs=-1)
    search.fit(X, y)

    print(f"best CV ROC AUC: {search.best_score_:.4f}")
    print("best params:")
    for k, v in search.best_params_.items():
        print(f"  {k.replace('clf__', '')}: {v}")


if __name__ == "__main__":
    main()

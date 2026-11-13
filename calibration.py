"""Probability calibration — make the predicted "X% default risk" trustworthy.

A Random Forest ranks borrowers well (high AUC) but its raw probabilities are not
guaranteed to mean what they say — a batch scored at 30% may not default 30% of the time.
OSFI Guideline E-23 expects model outputs to be sound, and in practice a probability of
default feeds IFRS 9 expected-credit-loss provisioning and risk-based pricing, so the
number has to be calibrated.

We wrap the model in an isotonic `CalibratedClassifierCV` and check it with the **Brier
score** and **Expected Calibration Error (ECE)**, plus a reliability curve.
"""

import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

import loan_default as L

REPORTS = L.REPORTS


def expected_calibration_error(y_true, proba, n_bins=10):
    """Mean gap between confidence and accuracy across probability bins."""
    y_true = np.asarray(y_true); proba = np.asarray(proba)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (proba >= lo) & (proba < hi)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - proba[mask].mean())
    return ece


def build_calibrated(base=None):
    """An isotonic-calibrated version of the Random Forest pipeline."""
    base = base or L.build_models()["Random Forest"]
    return CalibratedClassifierCV(base, method="isotonic", cv=3)


def _reliability_plot(y_test, raw, cal, path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    for proba, label, color in [(raw, "uncalibrated", "#dc2626"),
                                (cal, "calibrated", "#16a34a")]:
        frac, mean = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
        plt.plot(mean, frac, "o-", color=color, label=label)
    plt.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    plt.xlabel("predicted default probability"); plt.ylabel("observed default rate")
    plt.title("Reliability curve"); plt.legend(loc="upper left")
    plt.tight_layout(); os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=120); plt.close()


def main():
    df = L.engineer_features(L.load_data())
    X, y = df[L.RAW_CATEGORICAL + L.NUMERIC], df[L.TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42,
                                              stratify=y)
    raw_model = L.build_models()["Random Forest"].fit(X_tr, y_tr)
    cal_model = build_calibrated().fit(X_tr, y_tr)

    raw = raw_model.predict_proba(X_te)[:, 1]
    cal = cal_model.predict_proba(X_te)[:, 1]

    print(f"{'':<14}{'Brier':>8}{'ECE':>8}")
    print(f"{'uncalibrated':<14}{brier_score_loss(y_te, raw):>8.4f}"
          f"{expected_calibration_error(y_te, raw):>8.4f}")
    print(f"{'calibrated':<14}{brier_score_loss(y_te, cal):>8.4f}"
          f"{expected_calibration_error(y_te, cal):>8.4f}")

    _reliability_plot(y_te, raw, cal, os.path.join(REPORTS, "calibration_curve.png"))
    print(f"\nReliability curve -> {os.path.relpath(REPORTS, L.HERE)}/calibration_curve.png")


if __name__ == "__main__":
    main()

"""Fairness audit with Fairlearn — does the model treat protected groups evenly?

Lending in Canada is bound by the **Canadian Human Rights Act** (prohibited grounds
include age and sex) and OSFI Guideline E-23 expects models to be screened for bias. We
audit approval parity across **age** (a protected ground, real in the data) and
**province** (regional access). The dataset has no sex/gender field, so that protected
ground cannot be audited here — stated plainly rather than faked.

Uses Fairlearn's `MetricFrame` and the standard `demographic_parity_difference` /
`equalized_odds_difference` metrics instead of hand-rolled formulas. Reads
data/test_scored.csv + model_meta.json (written by train.py) — no retraining.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import pandas as pd
from fairlearn.metrics import (MetricFrame, count, selection_rate,
                               true_positive_rate, false_positive_rate,
                               demographic_parity_difference,
                               equalized_odds_difference)

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(HERE, "reports")
TEST_SCORED = os.path.join(HERE, "data", "test_scored.csv")
META = os.path.join(HERE, "model_meta.json")

AGE_BINS = [0, 25, 40, 60, 200]
AGE_LABELS = ["<=25", "26-40", "41-60", "60+"]


def load_scored():
    df = pd.read_csv(TEST_SCORED)
    df["age_band"] = pd.cut(df["person_age"], AGE_BINS, labels=AGE_LABELS)
    return df


def group_metrics(df, by, t):
    """Per-group fairness table at threshold ``t`` (decline when proba >= t)."""
    y_true = df["actual"]
    y_decline = (df["proba"] >= t).astype(int)   # the model's adverse action
    sens = df[by]

    mf = MetricFrame(
        metrics={"n": count, "decline_rate": selection_rate,
                 "recall": true_positive_rate, "fpr": false_positive_rate},
        y_true=y_true, y_pred=y_decline, sensitive_features=sens)

    table = mf.by_group.reset_index()
    table["approval_rate"] = 1 - table["decline_rate"]
    # Actual default rate per group (context for the decline rate).
    table = table.merge(
        df.assign(_d=y_true).groupby(by, observed=True)["_d"].mean()
          .rename("default_rate").reset_index(), on=by)
    top = table["approval_rate"].max()
    table["approval_ratio"] = (table["approval_rate"] / top).round(3)
    table["80%_rule"] = table["approval_ratio"].apply(lambda r: "ok" if r >= 0.8 else "FLAG")
    return table[[by, "n", "approval_rate", "default_rate", "recall", "fpr",
                  "approval_ratio", "80%_rule"]]


def disparities(df, by, t):
    """Fairlearn's scalar disparity metrics (0 = perfect parity)."""
    y_true = df["actual"]
    y_decline = (df["proba"] >= t).astype(int)
    return {
        "demographic_parity_difference":
            round(demographic_parity_difference(y_true, y_decline, sensitive_features=df[by]), 3),
        "equalized_odds_difference":
            round(equalized_odds_difference(y_true, y_decline, sensitive_features=df[by]), 3),
    }


def plot_group_metrics(table, by, path):
    import matplotlib.pyplot as plt
    x = range(len(table))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar([i - 0.2 for i in x], table["approval_rate"], width=0.4,
           label="approval rate", color="#16a34a")
    ax.bar([i + 0.2 for i in x], table["default_rate"], width=0.4,
           label="actual default rate", color="#dc2626")
    ax.set_xticks(list(x)); ax.set_xticklabels(table[by].astype(str))
    ax.set_ylabel("rate"); ax.set_title(f"Approval vs default rate by {by}"); ax.legend()
    plt.tight_layout(); os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=120); plt.close()


def main():
    with open(META, encoding="utf-8") as f:
        t = json.load(f)["threshold"]
    df = load_scored()
    print(f"Fairness at deployed threshold t = {t}  (Canadian Human Rights Act grounds)\n")

    for by, fname in [("age_band", "fairness_age.png"),
                      ("province", "fairness_province.png")]:
        table = group_metrics(df, by, t)
        print(f"== by {by} ==")
        print(table.round(3).to_string(index=False))
        print("  disparities:", disparities(df, by, t))
        flagged = table.loc[table["80%_rule"] == "FLAG", by].tolist()
        print("  four-fifths rule:",
              "all groups OK" if not flagged else f"FLAGGED: {flagged}", "\n")
        plot_group_metrics(table, by, os.path.join(REPORTS, fname))

    print(f"Charts -> {os.path.relpath(REPORTS, HERE)}/fairness_age.png, fairness_province.png")
    print("Note: the dataset has no sex/gender field, so that protected ground is not audited.")


if __name__ == "__main__":
    main()

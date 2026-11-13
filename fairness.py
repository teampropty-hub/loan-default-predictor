"""Fairness check: does the model treat age and home-ownership groups evenly?

Lending in the US is regulated (ECOA / Regulation B) against discrimination on
protected attributes. This dataset contains **age** (a protected attribute) and
**home ownership** (a wealth proxy, not legally protected but worth inspecting). It has
**no race or gender**, so those cannot be audited here — stated plainly rather than
faked.

For each group we report, at the deployed threshold: size, approval rate, actual default
rate, recall (share of real defaults caught), false-positive rate, and the approval-rate
ratio against the most-approved group. A ratio below 0.80 trips the classic
"four-fifths rule" used as a first screen for disparate impact.

Reads data/test_scored.csv + model_meta.json (produced by train.py) — no retraining.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # headless chart generation
import pandas as pd

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
    """Per-group fairness metrics at threshold ``t`` (decline when proba >= t)."""
    rows = []
    for name, g in df.groupby(by, observed=True):
        declined = g["proba"] >= t
        approved = ~declined
        defaulters = g["actual"] == 1
        goods = g["actual"] == 0
        rows.append({
            str(by): name,
            "n": len(g),
            "approval_rate": approved.mean(),
            "default_rate": defaulters.mean(),
            "recall": (declined & defaulters).sum() / max(defaulters.sum(), 1),
            "fpr": (declined & goods).sum() / max(goods.sum(), 1),
        })
    table = pd.DataFrame(rows)
    top = table["approval_rate"].max()
    table["approval_ratio"] = (table["approval_rate"] / top).round(3)
    table["80%_rule"] = table["approval_ratio"].apply(lambda r: "ok" if r >= 0.8 else "FLAG")
    return table


def plot_group_metrics(table, by, path):
    import matplotlib.pyplot as plt
    x = range(len(table))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar([i - 0.2 for i in x], table["approval_rate"], width=0.4,
           label="approval rate", color="#16a34a")
    ax.bar([i + 0.2 for i in x], table["default_rate"], width=0.4,
           label="actual default rate", color="#dc2626")
    ax.set_xticks(list(x)); ax.set_xticklabels(table[str(by)].astype(str))
    ax.set_ylabel("rate"); ax.set_title(f"Approval vs default rate by {by}")
    ax.legend()
    plt.tight_layout(); os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=120); plt.close()


def main():
    with open(META, encoding="utf-8") as f:
        t = json.load(f)["threshold"]
    df = load_scored()
    print(f"Fairness at deployed threshold t = {t}\n")

    for by, fname in [("age_band", "fairness_age.png"),
                      ("person_home_ownership", "fairness_home.png")]:
        table = group_metrics(df, by, t)
        print(f"== by {by} ==")
        print(table.round(3).to_string(index=False))
        flagged = table.loc[table["80%_rule"] == "FLAG", str(by)].tolist()
        print("  four-fifths rule:",
              "all groups OK" if not flagged else f"FLAGGED: {flagged}", "\n")
        plot_group_metrics(table, by, os.path.join(REPORTS, fname))

    print(f"Charts -> {os.path.relpath(REPORTS, HERE)}/fairness_age.png, fairness_home.png")
    print("\nNote: the dataset has no race or gender fields, so those cannot be audited here.")


if __name__ == "__main__":
    main()

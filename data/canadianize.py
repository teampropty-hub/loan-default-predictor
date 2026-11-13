"""Adapt the public credit-risk dataset to a Canadian framing (for demonstration).

No open dataset of individual Canadian loan defaults exists (PIPEDA), so this script takes
the public US-style Kaggle credit-risk data and reframes it for Canada:

  * money is treated as CAD (the distributions are plausible as Canadian dollars);
  * a `province` column is sampled from a realistic StatsCan population split, drawn
    INDEPENDENTLY of the outcome so no bias is fabricated;
  * a `credit_score` (300-900, Equifax/TransUnion-style) is DERIVED from the existing
    creditworthiness signals so it is realistically predictive.

This is a transparent demonstration proxy, not real Canadian bank data.

Run with:  python data/canadianize.py
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42

# Approximate share of the Canadian population by province (StatsCan, rounded).
PROVINCES = {
    "ON": 0.387, "QC": 0.225, "BC": 0.135, "AB": 0.117, "MB": 0.034,
    "SK": 0.029, "NS": 0.027, "NB": 0.022, "NL": 0.013, "PE": 0.004,
}
GRADE_POINTS = {"A": 1.0, "B": 0.83, "C": 0.66, "D": 0.5, "E": 0.33, "F": 0.16, "G": 0.0}


def add_province(df, rng):
    names = list(PROVINCES)
    weights = np.array(list(PROVINCES.values()))
    weights = weights / weights.sum()
    df["province"] = rng.choice(names, size=len(df), p=weights)
    return df


def add_credit_score(df, rng):
    """Derive a 300-900 score that moves with creditworthiness (so it is predictive)."""
    rate = df["loan_int_rate"].fillna(df["loan_int_rate"].median())
    grade = df["loan_grade"].map(GRADE_POINTS).fillna(0.5)
    no_prior_default = (df["cb_person_default_on_file"] == "N").astype(float)
    history = (df["cb_person_cred_hist_length"].fillna(0) / 20).clip(0, 1)
    low_rate = 1 - ((rate - 5) / 19).clip(0, 1)

    worthiness = 0.55 * grade + 0.20 * no_prior_default + 0.15 * history + 0.10 * low_rate
    noise = rng.normal(0, 0.04, size=len(df))
    score = 300 + 600 * (worthiness + noise).clip(0, 1)
    df["credit_score"] = score.round().clip(300, 900).astype(int)
    return df


def canadianize(src, dst):
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(os.path.join(HERE, src))
    df = add_province(df, rng)
    df = add_credit_score(df, rng)
    df.to_csv(os.path.join(HERE, dst), index=False)
    print(f"{src} -> {dst}  ({len(df):,} rows, +province +credit_score)")
    return df


def main():
    train = canadianize("credit_risk_train.csv", "ca_credit_train.csv")
    canadianize("loan_requests.csv", "ca_loan_requests.csv")
    print("\ncredit_score summary:\n" + train["credit_score"].describe().round(0).to_string())
    print("\nprovince split:\n" + (train["province"].value_counts(normalize=True)
                                   .round(3)).to_string())


if __name__ == "__main__":
    main()

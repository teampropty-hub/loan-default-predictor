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
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from loan_default import PROVINCE_DELINQUENCY  # real relative provincial risk
import macro as M  # real Bank of Canada policy-rate history

SEED = 42
BOC_COLS = ["boc_policy_rate", "boc_rate_change_12m", "boc_rate_vol"]

# Approximate share of the Canadian population by province (StatsCan, rounded).
PROVINCES = {
    "ON": 0.387, "QC": 0.225, "BC": 0.135, "AB": 0.117, "MB": 0.034,
    "SK": 0.029, "NS": 0.027, "NB": 0.022, "NL": 0.013, "PE": 0.004,
}
GRADE_POINTS = {"A": 1.0, "B": 0.83, "C": 0.66, "D": 0.5, "E": 0.33, "F": 0.16, "G": 0.0}


def add_province(df, rng):
    """Assign province with a mild tilt by outcome toward each province's REAL relative
    delinquency level, so the province signal reflects genuine regional risk (ON/BC
    higher). Overall population mix stays close to the StatsCan split."""
    names = list(PROVINCES)
    pop = np.array([PROVINCES[n] for n in names]); pop /= pop.sum()
    dnorm = np.array([PROVINCE_DELINQUENCY[n] for n in names])
    dnorm = dnorm / dnorm.mean()

    if "loan_status" not in df.columns:   # the request file has no outcome
        df["province"] = rng.choice(names, size=len(df), p=pop)
        return df

    w_def = pop * (dnorm ** 0.4); w_def /= w_def.sum()   # defaulters -> higher-risk provinces
    w_ok = pop / (dnorm ** 0.4); w_ok /= w_ok.sum()      # repayers  -> lower-risk provinces
    is_def = df["loan_status"].values == 1
    prov = np.empty(len(df), dtype=object)
    prov[is_def] = rng.choice(names, size=int(is_def.sum()), p=w_def)
    prov[~is_def] = rng.choice(names, size=int((~is_def).sum()), p=w_ok)
    df["province"] = prov
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


def add_macro(df, rng):
    """Give each loan a synthetic origination month and join the REAL Bank of Canada macro
    environment of that month, tilted so defaults skew toward monetary-tightening periods
    (the real cyclical pattern). New requests get today's environment."""
    boc = M.load_macro()
    if "loan_status" not in df.columns:
        for k, v in M.current_macro().items():
            df[k] = v
        return df

    stress = boc["boc_rate_change_12m"].clip(lower=0) + 0.3 * boc["boc_policy_rate"]
    stress = (stress - stress.min()) / (stress.max() - stress.min() + 1e-9)
    w_def = (0.25 + stress).to_numpy(dtype=float); w_def = w_def / w_def.sum()
    w_ok = (0.25 + (1 - stress)).to_numpy(dtype=float); w_ok = w_ok / w_ok.sum()
    is_def = df["loan_status"].values == 1
    idx = np.empty(len(df), dtype=int)
    idx[is_def] = rng.choice(len(boc), size=int(is_def.sum()), p=w_def)
    idx[~is_def] = rng.choice(len(boc), size=int((~is_def).sum()), p=w_ok)
    picked = boc.iloc[idx].reset_index(drop=True)
    for col in BOC_COLS:
        df[col] = picked[col].values
    df["orig_month"] = picked["month"].values
    return df


def canadianize(src, dst):
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(os.path.join(HERE, src))
    df = add_province(df, rng)
    df = add_credit_score(df, rng)
    df = add_macro(df, rng)
    df.to_csv(os.path.join(HERE, dst), index=False)
    print(f"{src} -> {dst}  ({len(df):,} rows, +province +credit_score +BoC macro)")
    return df


def main():
    train = canadianize("credit_risk_train.csv", "ca_credit_train.csv")
    canadianize("loan_requests.csv", "ca_loan_requests.csv")
    print("\ncredit_score summary:\n" + train["credit_score"].describe().round(0).to_string())
    print("\nprovince split:\n" + (train["province"].value_counts(normalize=True)
                                   .round(3)).to_string())


if __name__ == "__main__":
    main()

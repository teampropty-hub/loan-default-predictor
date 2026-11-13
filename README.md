# 💳 Loan Default Predictor

> A supervised **machine-learning** project that predicts the probability a loan will
> default, then wraps the model in the tooling a real credit team needs: a
> cost-optimal decision cutoff, a fairness audit, and an interactive demo.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-demo-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**ROC AUC 0.93 · catches 75% of real defaults · cost-optimal threshold saves ~$115k vs a naive 0.5.**

**→ [`analysis.ipynb`](./analysis.ipynb) for the full walkthrough · `streamlit run app.py` for the live demo.**

![Demo](reports/demo.png)

---

## What it does

Given a loan application (age, income, home ownership, loan amount, interest rate,
credit history…), the model outputs a **default probability** and a decision —
**approve / review / decline**. It's trained on ~24,000 historical loans where the
outcome (repaid or defaulted) is known.

This is **binary classification**: the model *learns* the patterns that separate good
loans from bad ones, instead of following hand-written rules.

## Quickstart

```bash
pip install -r requirements.txt
python train.py             # train, pick the threshold, write the model + charts
streamlit run app.py        # interactive demo (score applicants, move cost sliders)
```

Standalone scripts: `python loan_default.py` (scorecard) · `python threshold.py`
(cost curve) · `python fairness.py` (group audit).

## How it works

1. **Feature engineering** — on top of the raw columns, add interpretable ratios a
   credit analyst would actually look at: `loan_to_income`, `credit_history_ratio`,
   `income_per_age`; cap impossible outliers (employment length > 47 years).
2. **Model comparison** — Logistic Regression vs Random Forest vs Gradient Boosting,
   ranked by **5-fold cross-validated ROC AUC** inside a leakage-safe scikit-learn
   `Pipeline` (encoding + scaling fit on training data only).
3. **Cost-based threshold** — replace the arbitrary 0.5 cutoff with the one that
   minimises expected money lost.
4. **Fairness audit** — check approval parity across age and home-ownership groups.
5. **Interactive demo** — a Streamlit app to score applicants and explore the trade-offs.

## Results

5-fold CV ROC AUC: **Random Forest 0.929**, Gradient Boosting 0.928, Logistic 0.877.
Held-out test set (20% never seen in training):

| model | accuracy | precision | recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 82.3% | 56.5% | 78.6% | 65.7% | 0.879 |
| **Random Forest** ⭐ | **92.7%** | **89.1%** | **75.5%** | **81.7%** | **0.933** |
| Gradient Boosting | 92.8% | 94.5% | 71.1% | 81.1% | 0.932 |

| ROC curves | Feature importance | Confusion matrix |
|---|---|---|
| ![ROC](reports/roc_curves.png) | ![Importance](reports/feature_importance.png) | ![Confusion](reports/confusion_matrix.png) |

> With ~22% defaults the classes are imbalanced, so **ROC AUC and recall** matter more
> than raw accuracy — the goal is to catch real defaults without drowning good
> borrowers in false declines.

## Cost-based threshold

A 0.5 cutoff is arbitrary. The two mistakes cost very different amounts: approving a
loan that defaults loses `principal × LGD`, while declining a good borrower loses the
interest we would have earned (`principal × rate × term`). Minimising total expected
cost on the held-out set moves the threshold to **0.58**, saving **~$115k** versus 0.5
(at LGD 60%, 3-year term). The demo lets you move those assumptions and watch the
threshold shift live.

![Cost vs threshold](reports/cost_threshold.png)

## Fairness

Lending is regulated (ECOA / Regulation B) against discrimination. The dataset contains
**age** (a protected attribute) and **home ownership** (a wealth proxy); it has **no
race or gender**, so those cannot be audited — stated plainly rather than faked.

Approval parity by the four-fifths rule: **age groups pass** (all within 6% of each
other), but **renters are flagged** — their approval-rate ratio is ~0.78 (< 0.80),
driven by a genuinely higher default rate (31% vs ~12% for mortgage holders). That is
the accuracy-vs-fairness tension a real credit team has to decide how to handle — the
project surfaces it rather than hiding it.

![Fairness by group](reports/fairness_home.png)

## Repo map

| File | Purpose |
|---|---|
| `loan_default.py` | Data loading, feature engineering, models, scoring (shared core) |
| `train.py` | Trains the model, picks the threshold, writes `model.joblib` + charts |
| `threshold.py` | Cost-minimising decision threshold |
| `fairness.py` | Group fairness metrics (four-fifths rule) |
| `app.py` | Streamlit demo |
| `analysis.ipynb` | Narrative walkthrough with charts |
| `data/` | Credit-risk dataset + pending loan requests |
| `reports/` | Generated charts |

## What this project demonstrates

- End-to-end supervised ML: feature engineering → model comparison → evaluation.
- Leakage-safe pipelines and **honest validation** (cross-validation, held-out test).
- Picking the **right metric** for imbalanced data (AUC/recall over accuracy).
- Turning model scores into a **business decision** via cost analysis.
- **Responsible ML**: a fairness screen, with honest limits about the data.

## Data & disclaimer

`data/` holds a public-style credit-risk dataset used purely for demonstration. This is
a **learning project, not a production credit model** — the fairness screen is a first
look, not a compliance sign-off.

## License

[MIT](./LICENSE)
